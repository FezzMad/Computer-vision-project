import os
from itertools import islice
from pathlib import Path


def split_by_weights(list_for_split, weights):
    train, val, test = weights

    size = len(list_for_split)
    train_size = int(size * train)
    val_size = int(size * val)
    test_size = size - train_size - val_size

    size_split = [train_size, val_size, test_size]
    inter_list = iter(list_for_split)
    train_list, val_list, test_list = [list(islice(inter_list, elem)) for elem in size_split]

    return train_list, val_list, test_list


def mk_folder(folder):
    if not os.path.exists(folder):
        Path(folder).mkdir(parents=True, exist_ok=True)


def cp_file(src, dest):
    if os.path.isfile(src) and os.path.isdir(dest):
        file_name = os.path.normpath(src).split(os.sep)[-1]
        new_path = f'{dest}/{file_name}'
        if not os.path.exists(new_path):
            os.system(f"cp {src} {dest}")


imgs_path = "/home/user/Desktop/my_nn/data/apples_compressed"
dataset_name = "dataset"

train = 0.7
val = 0.2
test = 0.1

dataset_path = f'{"/".join(os.path.normpath(imgs_path).split(os.sep)[:-1])}/{dataset_name}'
train_path = f'{dataset_path}/train'
val_path = f'{dataset_path}/val'
test_path = f'{dataset_path}/test'


for subdir, dirs, files in os.walk(imgs_path):
    if files:
        class_name = os.path.normpath(os.path.relpath(subdir, imgs_path)).split(os.sep)[0]

        sub_train_path = f'{train_path}/{class_name}'
        sub_val_path = f'{val_path}/{class_name}'
        sub_test_path = f'{test_path}/{class_name}'

        mk_folder(f'{train_path}/{class_name}')
        mk_folder(f'{val_path}/{class_name}')
        mk_folder(f'{test_path}/{class_name}')

        # files may be shuffled for random
        path = list(map(lambda file: f'{subdir}/{file}', files))
        train_list, val_list, test_list = split_by_weights(path, [train, val, test])

        for img in train_list:
            cp_file(img, sub_train_path)
        for img in val_list:
            cp_file(img, sub_val_path)
        for img in test_list:
            cp_file(img, sub_test_path)


