def normal_time(seconds):
    """
    Converting seconds in float to str in format: 'Xh Xm Xs', 'Xm Xs', 'Xh', 'Xm', 'Xs', 'Xh Xs', 'Xh Xm'

    :param seconds: (float) seconds in float type
    :return: (str) time in format: 'Xh Xm Xs', 'Xm, Xs', 'Xh', 'Xm', 'Xs', 'Xh Xs', 'Xh Xm'
    """
    from datetime import timedelta

    h, m, s = map(lambda x: int(x), str(timedelta(seconds=int(seconds))).split(":"))
    norm_time = ""
    if h and m and s:
        norm_time = f'{h}h {m}m {s}s'
    elif h and m:
        norm_time = f'{h}h {m}m'
    elif h and s:
        norm_time = f'{h}h {s}s'
    elif m and s:
        norm_time = f'{m}m {s}s'
    elif s:
        norm_time = f'{s}s'
    elif m:
        norm_time = f'{m}m'
    elif h:
        norm_time = f'{h}h'
    return norm_time


def mk_dir(save_path):
    """
    Creating a target folder and parent folders, if necessary

    :param save_path: (str) path to target folder
    :return: (str) path to target folder
    """
    from pathlib import Path

    Path(save_path).mkdir(parents=True, exist_ok=True)
    return save_path


def mk_training_dir(save_path):
    """
    Creating a folder on the target path to save the training results

    :param save_path: (str) target path
    :return: (str) path to training folder
    """
    import os

    experiment = 'train1'
    train_folder = os.path.join(save_path, experiment)
    if not os.path.exists(train_folder):
        mk_dir(train_folder)
    else:
        while os.path.exists(train_folder):
            save_path, experiment = os.path.split(train_folder)
            experiment = experiment[:5] + str(int(experiment[5:]) + 1)
            train_folder = os.path.join(save_path, experiment)
        mk_dir(train_folder)
    return train_folder


def save_list(list_name, save_path, file_name):
    """
    Saving list in .npy format

    :param list_name: (list) list for saving
    :param save_path: (str) path to saving
    :param file_name: (str) name of the saved file
    :return: None
    """
    import numpy as np

    save_name = f'{save_path}/{file_name}.npy'
    np.save(save_name, list_name)


def load_list(path):
    """
    Loading .npy file as a list

    :param path: (str) full name of the file with the .npy extension
    :return: (list) loaded file as a list
    """
    import numpy as np

    # TODO add an extension check
    temp_numpy_array = np.load(path)
    return temp_numpy_array.tolist()


def plot_graphs(training, validation, save_path, file_name):
    """
    Plotting and saving loss and accuracy graphs for training and validation

    :param training: (list) list with values from training with the size of the number of epochs
    :param validation: (list) list with values from validation with the size of the number of epochs
    :param save_path: (str) path to saving
    :param file_name: (str) name of the saved file
    :return: None
    """
    from matplotlib.ticker import FormatStrFormatter
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(training, label='training')
    ax.plot(validation, label='validation')
    ax.legend()
    ax.set_xlabel('epochs')
    ax.set_ylabel(file_name)
    ax.set(xlim=(0, len(training)))
    ax.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    plt.savefig(f'{save_path}/{file_name}.jpg')
    plt.clf()


def plot_confusion_matrix(matrix, labels, save_path):
    """
    Plotting and saving confusion matrix

    :param matrix: (numpy.ndarray) matrix values
    :param labels: (list) labels names for displaying
    :param save_path: (str) path to saving
    :return: None
    """
    from sklearn.metrics import ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    disp = ConfusionMatrixDisplay(confusion_matrix=matrix,
                                  display_labels=labels)
    fig, ax = plt.subplots(figsize=(9, 7))
    disp.plot(cmap=plt.cm.Greens, ax=ax)
    disp.figure_.savefig(f'{save_path}/confusion_matrix.jpg')
