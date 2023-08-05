import os
import torch
import torchvision
import random
import numpy as np

IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG',
                  '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return sorted(images)


def augment(img_list, hflip=True, rot=True, split='val'):
    # horizontal flip OR rotate
    hflip = hflip and (split == 'train' and random.random() < 0.5)
    vflip = rot and (split == 'train' and random.random() < 0.5)
    rot90 = rot and (split == 'train' and random.random() < 0.5)

    def _augment(img):
        if hflip:
            img = img[:, ::-1, :]
        if vflip:
            img = img[::-1, :, :]
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    return [_augment(img) for img in img_list]


def transform2numpy(img):
    img = np.array(img)
    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img


def transform2tensor(img, min_max=(0, 1)):
    # HWC to CHW
    img = torch.from_numpy(np.ascontiguousarray(
        np.transpose(img, (2, 0, 1)))).float()
    # to range min_max
    img = img * (min_max[1] - min_max[0]) + min_max[0]
    return img


def random_crop(stacked_img, patch_size):
    # img_size: int
    # stacked_image shape: 2*C*H*W type: tensor
    h, w = stacked_img.shape[2], stacked_img.shape[3]
    start_h = np.random.randint(low=0, high=(h - patch_size) + 1) if h > patch_size else 0
    start_w = np.random.randint(low=0, high=(h - patch_size) + 1) if w > patch_size else 0
    return stacked_img[:, :, start_h:start_h + patch_size, start_w:start_w + patch_size]


# implementation by numpy and torch
# def transform_augment(img_list, split='val', min_max=(0, 1)):
#     imgs = [transform2numpy(img) for img in img_list]
#     imgs = augment(imgs, split=split)
#     ret_img = [transform2tensor(img, min_max) for img in imgs]
#     return ret_img


# implementation by torchvision, detail in https://github.com/Janspiry/Image-Super-Resolution-via-Iterative-Refinement/issues/14
totensor = torchvision.transforms.ToTensor()
hflip = torchvision.transforms.RandomHorizontalFlip()
preresize = torchvision.transforms.Resize([256, 256])


def transform_augment(img_list, split='val', min_max=(0, 1), patch_size=160):
    imgs = [preresize(img) for img in img_list]
    imgs = [totensor(img) for img in imgs]
    img_mask = imgs[-1]
    img_mask = img_mask.repeat(3, 1, 1)
    imgs[-1] = img_mask
    imgs = torch.stack(imgs, 0)
    if split == 'train':
        imgs = random_crop(imgs, patch_size=patch_size)
        imgs = hflip(imgs)
    crop_h, crop_w = imgs.shape[2] % 16, imgs.shape[3] % 16
    imgs = imgs[:, :, :imgs.shape[2] - crop_h, :imgs.shape[3] - crop_w]
    imgs = torch.unbind(imgs, dim=0)

    ret_img = [img * (min_max[1] - min_max[0]) + min_max[0] for img in imgs[0:-1]]
    ret_img.append(imgs[-1])
    ret_img[-1] = torch.mean(ret_img[-1], 0, keepdim=True)
    return ret_img
