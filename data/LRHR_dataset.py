from io import BytesIO
import lmdb
from PIL import Image
from torch.utils.data import Dataset
import random
import data.util as Util
import os


class LRHRDataset(Dataset):
    def __init__(self, dataroot, datatype, l_resolution=16, r_resolution=128, split='train', data_len=-1, need_LR=False):
        self.datatype = datatype
        self.l_res = l_resolution
        self.r_res = r_resolution
        self.data_len = data_len
        self.need_LR = need_LR
        self.split = split
        if split == 'train':
            gt_dir = 'train_C'
            input_dir = 'train_A'
            mask_dir = 'train_B'
            # gt_dir = 'Normal'
            # input_dir = 'Low'
        else:
            gt_dir = 'test_C'
            input_dir = 'test_A'
            mask_dir = 'test_B'
            # gt_dir = 'Normal'
            # input_dir = 'Low'

        if datatype == 'lmdb':
            self.env = lmdb.open(dataroot, readonly=True, lock=False,
                                 readahead=False, meminit=False)
            # init the datalen
            with self.env.begin(write=False) as txn:
                self.dataset_len = int(txn.get("length".encode("utf-8")))
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        elif datatype == 'img':
            clean_files = sorted(os.listdir(os.path.join(dataroot, gt_dir)))
            noisy_files = sorted(os.listdir(os.path.join(dataroot, input_dir)))
            mask_files = sorted(os.listdir(os.path.join(dataroot, mask_dir)))

            self.hr_path = [os.path.join(dataroot, gt_dir, x) for x in clean_files]
            self.sr_path = [os.path.join(dataroot, input_dir, x) for x in noisy_files]
            self.mask_path = [os.path.join(dataroot, mask_dir, x) for x in mask_files]
            # self.sr_path = Util.get_paths_from_images(
            #     '{}/sr_{}_{}'.format(dataroot, l_resolution, r_resolution))
            # self.hr_path = Util.get_paths_from_images(
            #     '{}/hr_{}'.format(dataroot, r_resolution))
            # if self.need_LR:
            #     self.lr_path = Util.get_paths_from_images(
            #         '{}/lr_{}'.format(dataroot, l_resolution))
            self.dataset_len = len(self.hr_path)
            if self.data_len <= 0:
                self.data_len = self.dataset_len
            else:
                self.data_len = min(self.data_len, self.dataset_len)
        else:
            raise NotImplementedError(
                'data_type [{:s}] is not recognized.'.format(datatype))

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        img_HR = None
        img_LR = None

        if self.datatype == 'lmdb':
            with self.env.begin(write=False) as txn:
                hr_img_bytes = txn.get(
                    'hr_{}_{}'.format(
                        self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                sr_img_bytes = txn.get(
                    'sr_{}_{}_{}'.format(
                        self.l_res, self.r_res, str(index).zfill(5)).encode('utf-8')
                )
                if self.need_LR:
                    lr_img_bytes = txn.get(
                        'lr_{}_{}'.format(
                            self.l_res, str(index).zfill(5)).encode('utf-8')
                    )
                # skip the invalid index
                while (hr_img_bytes is None) or (sr_img_bytes is None):
                    new_index = random.randint(0, self.data_len-1)
                    hr_img_bytes = txn.get(
                        'hr_{}_{}'.format(
                            self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    sr_img_bytes = txn.get(
                        'sr_{}_{}_{}'.format(
                            self.l_res, self.r_res, str(new_index).zfill(5)).encode('utf-8')
                    )
                    if self.need_LR:
                        lr_img_bytes = txn.get(
                            'lr_{}_{}'.format(
                                self.l_res, str(new_index).zfill(5)).encode('utf-8')
                        )
                img_HR = Image.open(BytesIO(hr_img_bytes)).convert("RGB")
                img_SR = Image.open(BytesIO(sr_img_bytes)).convert("RGB")
                if self.need_LR:
                    img_LR = Image.open(BytesIO(lr_img_bytes)).convert("RGB")
        else:

            img_SR = Image.open(self.sr_path[index]).convert("RGB")
            if self.split == 'train':
                hr_name = self.sr_path[index].replace('.jpg', '_no_shadow.jpg')
            else:
                hr_name = self.sr_path[index].replace('.jpg', '_free.jpg')
            hr_name = hr_name.replace('_A', '_C')
            img_HR = Image.open(hr_name).convert("RGB")
            img_mask = Image.open(self.mask_path[index]).convert("1")
            if self.need_LR:
                img_LR = Image.open(self.sr_path[index]).convert("RGB")
        if self.need_LR:
            [img_LR, img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_LR, img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'LR': img_LR, 'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index}
        else:
            [img_SR, img_HR, img_mask] = Util.transform_augment(
                [img_SR, img_HR, img_mask], split=self.split, min_max=(-1, 1))
            return {'HR': img_HR, 'SR': img_SR, 'mask': img_mask, 'Index': index}
