import torch
import numpy as np
import os
import cv2

from data.utils import img_preparation


class Dataset:
    def __init__(self, path, imgsz):
        self.path = path
        self._train_path = f'{path}/train/'
        self._test_path = f'{path}/test/'
        self._val_path = f'{path}/val/'

        self.val = Dataset2Class(self._val_path, imgsz)
        self.test = Dataset2Class(self._test_path, imgsz)
        self.train = Dataset2Class(self._train_path, imgsz)

        self.classes = self.train.classes
        self.num_classes = self._get_num_classes()

    def __len__(self):
        return len(self.train) + len(self.test) + len(self.val)

    def _get_num_classes(self):
        return len(self.classes)


class Dataset2Class(torch.utils.data.Dataset):
    def __init__(self, path, imgsz):
        super().__init__()
        self.path = path
        self._imgsz = imgsz

        self.classes = self._get_classes()
        self.num_classes = self._get_num_classes()

        self._data_list = self._get_data_list()

    def _get_classes(self):
        return {class_name: idx for idx, class_name in
                enumerate(sorted(os.listdir(self.path)))}  # dictionary of index and class name

    def _get_data_list(self):
        classes_folders = sorted(os.listdir(self.path))
        data_list = []
        for cls_folder in classes_folders:
            class_path = f'{self.path}/{cls_folder}/'
            class_list = sorted(os.listdir(class_path))
            data_list += list((os.path.join(class_path, data), self.classes[cls_folder]) for data in class_list)

        return data_list

    def _get_num_classes(self):
        return len(self.classes)

    def __len__(self):
        return len(self._data_list)

    def __getitem__(self, idx):
        img_path = self._data_list[idx][0]
        class_id = self._data_list[idx][1]

        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = img_preparation(img, self._imgsz)

        t_img = torch.from_numpy(img)
        t_class_id = torch.tensor(class_id)

        return {"img": t_img, "label": t_class_id}
