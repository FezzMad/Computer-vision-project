import cv2
import os
from pathlib import Path
from tqdm import tqdm


def compress_image(root_path, save_path, imgsz, quality):
    if not os.path.exists(save_path):
        img = cv2.imread(root_path)
        img_shape = img.shape
        if img_shape[0] > imgsz or img_shape[1] > imgsz:
            img = cv2.resize(img, (imgsz, imgsz), cv2.INTER_AREA)
        cv2.imwrite(save_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])


def get_images(path):
    imgs_list = []
    for x in path.rglob("*.jpg"):
        img_path = os.path.relpath(x, path)
        imgs_list.append([str(x), img_path])
    return imgs_list


def mk_folders(imgs, save_root):
    for img in imgs:
        folder_path = "/".join(os.path.normpath(img[1]).split(os.sep)[:-1])
        save_path = f'{save_root}/{folder_path}'
        if not os.path.exists(save_path):
            Path(save_path).mkdir(parents=True, exist_ok=True)


imgsz = 640
jpeg_quality = 40
dataset_path = Path('/home/user/Desktop/my_nn/data/apples')
save_root = f'{dataset_path}_compressed'


imgs_list = get_images(dataset_path)
mk_folders(imgs_list, save_root)
for img_path in (pbar := tqdm(imgs_list)):
    compress_image(img_path[0], f'{save_root}/{img_path[1]}', imgsz, jpeg_quality)
    pbar.set_description(f'Compressing images')







