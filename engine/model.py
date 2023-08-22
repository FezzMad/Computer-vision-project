import os
import yaml
import time
import logging.config
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.nn.functional as F
from tqdm import tqdm

from data.dataset import Dataset
from nn.architecture import YoloClassifySmall, MyNet
from nn.utils import save_model
from utils.utils import mk_training_dir, save_list, plot_graphs, normal_time, mk_dir, plot_confusion_matrix
from utils.metrics import accuracy


class NNModel:
    def __init__(self, yaml_cfg):
        self.cfg = self._load_configurations(yaml_cfg)

        self._dataset = self._dataset()
        self.nc = self._dataset.num_classes
        self.architecture = self._architecture()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.amp = self.cfg['amp']

        self._save_path = self.cfg['save_path']

        self._train_loss_epochs_list = []
        self._train_acc_epochs_list = []
        self._valid_loss_epochs_list = []
        self._valid_acc_epochs_list = []

        self._logger = None

        self._lists_save_path = None
        self._graphs_save_path = None

    def train(self):
        epochs = self.cfg['epochs']

        model = self.architecture.to(self.device)

        optimizer = self._optimizer(model.parameters())

        loss_fn = torch.nn.CrossEntropyLoss()
        loss_fn = loss_fn.to(self.device)

        scaler = GradScaler()  # for AMP

        torch.backends.cudnn.benchmark = True  # to select the optimal algorithm for a fixed tensor
        torch.backends.cudnn.deterministic = True  # for reproducibility of results

        start_time = time.time()

        self._save_path = mk_training_dir(self._save_path)

        logging.basicConfig(filename=f'{self._save_path}/log.txt',
                                          filemode='a')
        LOGGER = logging.getLogger()
        self._logger = LOGGER

        train_loader = self._loader(self._dataset.train)
        valid_loader = self._loader(self._dataset.val)
        test_loader = self._loader(self._dataset.test)

        for epoch in range(epochs):
            self._train(model, train_loader, loss_fn, optimizer, scaler, epoch, epochs)
            self._validate(model, valid_loader, loss_fn)
            save_model(model, self._save_path)

        self._graphs_save_path = mk_dir(f'{self._save_path}/graphs')
        self._lists_save_path = mk_dir(f'{self._save_path}/graphs_arrays')

        self._test(model, test_loader)

        save_list(self._train_loss_epochs_list, self._lists_save_path, 'train_loss')
        save_list(self._train_acc_epochs_list, self._lists_save_path, 'train_accuracy')
        save_list(self._valid_loss_epochs_list, self._lists_save_path, 'valid_loss')
        save_list(self._valid_acc_epochs_list, self._lists_save_path, 'valid_accuracy')
        self._save_configurations(self.cfg, self._save_path)

        plot_graphs(self._train_loss_epochs_list, self._valid_loss_epochs_list, self._graphs_save_path, "loss")
        plot_graphs(self._train_acc_epochs_list, self._valid_acc_epochs_list, self._graphs_save_path, "accuracy")

        stop_time = time.time()
        cycle_time = normal_time(stop_time - start_time)

        LOGGER.warning(f'Training time: {cycle_time}')

    def _test(self, model, loader):
        from sklearn.metrics import confusion_matrix
        import numpy as np

        conf_matr = []
        model.eval()
        labels = [x for x in range(self.nc)]
        with torch.no_grad():
            for sample in (pbar := tqdm(loader)):
                img, label = sample['img'], sample['label']
                img = img.to(self.device)
                label = label.to(self.device)
                label = F.one_hot(label, self.nc).float()

                pred = model(img)

                y_true = label.cpu().float().numpy().argmax(1)
                y_pred = F.softmax(pred.detach().cpu().float(), dim=1).numpy().argmax(1)

                if not len(conf_matr):
                    conf_matr = confusion_matrix(y_true, y_pred, labels=labels)
                else:
                    conf_matr = np.add(conf_matr, confusion_matrix(y_true, y_pred, labels=labels))

                pbar.set_description(f'Testing model')

        save_list(list(conf_matr), self._lists_save_path, 'confusion_matrix')
        labels_names = [labels for labels, idx in self._dataset.classes.items()]
        plot_confusion_matrix(conf_matr, labels_names, self._graphs_save_path)

    def _train(self, model, loader, loss_fn, optimizer, scaler, epoch, epochs):
        size = len(loader)
        sample_loss = 0.0
        sample_accuracy = 0.0
        model.train()
        for sample in (pbar := tqdm(loader)):
            img, label = sample['img'], sample['label']
            img = img.to(self.device)
            label = label.to(self.device)
            label = F.one_hot(label, self.nc).float()

            with autocast(self.amp):
                pred = model(img)
                loss = loss_fn(pred, label)

            scaler.scale(loss).backward()
            running_loss = loss.item()
            sample_loss += running_loss
            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad()

            y_true = label.cpu().float().numpy().argmax(1)
            y_pred = F.softmax(pred.detach().cpu().float(), dim=1).numpy().argmax(1)

            running_accuracy = accuracy(y_true, y_pred)
            sample_accuracy += running_accuracy
            pbar.set_description(
                f'{epoch + 1}/{epochs}| Loss: {running_loss:.3f}\tAccuracy: {running_accuracy:.3f}')

        train_loss = sample_loss / size
        train_accuracy = sample_accuracy / size
        self._train_loss_epochs_list.append(train_loss)
        self._train_acc_epochs_list.append(train_accuracy)

    def _validate(self, model, loader, loss_fn):
        size = len(loader)
        sample_loss = 0.0
        sample_accuracy = 0.0
        model.eval()
        with torch.no_grad():
            for sample in tqdm(loader):
                img, label = sample['img'], sample['label']
                img = img.to(self.device)
                label = label.to(self.device)
                label = F.one_hot(label, self.nc).float()

                pred = model(img)
                loss = loss_fn(pred, label)
                running_loss = loss.item()
                sample_loss += running_loss

                y_true = label.cpu().float().numpy().argmax(1)
                y_pred = F.softmax(pred.detach().cpu().float(), dim=1).numpy().argmax(1)

                running_accuracy = accuracy(y_true, y_pred)
                sample_accuracy += running_accuracy

        valid_loss = sample_loss / size
        valid_accuracy = sample_accuracy / size
        self._valid_loss_epochs_list.append(valid_loss)
        self._valid_acc_epochs_list.append(valid_accuracy)

    def _loader(self, dataset):
        batch_size = self.cfg['batch']
        return torch.utils.data.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=2,
                                           pin_memory=False)

    def _optimizer(self, parametres):
        name = self.cfg['optimizer']
        lr = self.cfg['lr']
        momentum = self.cfg['momentum']
        weight_decay = self.cfg['weight_decay']
        beta2 = self.cfg['beta2']
        if name == 'Adam':
            optimizer = torch.optim.Adam(parametres, lr=lr, betas=(momentum, beta2))  # adjust beta1 to momentum
        elif name == 'AdamW':
            optimizer = torch.optim.AdamW(parametres, lr=lr, betas=(momentum, beta2), weight_decay=weight_decay)
        elif name == 'RMSProp':
            optimizer = torch.optim.RMSprop(parametres, lr=lr, momentum=momentum)
        elif name == 'SGD':
            optimizer = torch.optim.SGD(parametres, lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(f'Optimizer {name} not implemented.')
        return optimizer

    def _architecture(self):
        architecture = self.cfg['architecture']
        if architecture == 'YoloNetC':
            return YoloClassifySmall(self.nc)
        elif architecture == 'MyNet1':
            return MyNet(self.nc)

    def _dataset(self):
        path_dataset = self.cfg['data']  # path to three folders: "test", "train", "val"
        imgsz = self.cfg['imgsz']
        dataset = Dataset(path_dataset, imgsz)
        return dataset

    def _classes(self):
        return [key for key, value in self._dataset.classes.items()]

    @staticmethod
    def _save_configurations(dict_cfg, save_path):
        """
        Saving parameters from the configuration directory to a file with the extension .yaml

        :param dict_cfg: (str): directory with configurations
        :param save_path: (str): path to save the .yaml file
        :return:
        """

        with open(f'{save_path}/cfg.yaml', 'w') as file:
            yaml.dump(dict_cfg, file)

    @staticmethod
    def _load_configurations(yaml_cfg):
        """
        Loading parameters from a configuration file with the extension .yaml to dictionary

        :param yaml_cfg: (str) the name of the configuration file with the extension .yaml from the "cfg" folder of
        this project
        :return: (dict): YAML data in dictionary
        """

        root_path = os.getcwd()
        root_yaml_path = f'{root_path}/cfg'
        yaml_path = f'{root_yaml_path}/{yaml_cfg}'

        cfg = None
        if os.path.exists(yaml_path):
            with open(yaml_path, 'r') as file_option:
                cfg = yaml.safe_load(file_option)
        else:
            pass
            # LOGGER.warning(f"WARNING ⚠️ configuration file not found: {yaml_path}")
        return cfg
