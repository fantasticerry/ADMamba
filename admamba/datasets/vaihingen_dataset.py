import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
import matplotlib.pyplot as plt
import albumentations as albu

import matplotlib.patches as mpatches
from PIL import Image
import random
from .transform import *

CLASSES = ('ImSurf', 'Building', 'LowVeg', 'Tree', 'Car', 'Clutter')
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 255], [0, 255, 0], [255, 204, 0], [255, 0, 0]]

ORIGIN_IMG_SIZE = (1024, 1024)
INPUT_IMG_SIZE = (1024, 1024)
TEST_IMG_SIZE = (1024, 1024)


def get_training_transform():
    train_transform = [
        albu.RandomRotate90(p=0.5),
        albu.Normalize()
    ]
    return albu.Compose(train_transform)

def get_training_transform_geom_only():
    """只包含几何变换，不归一化，用于处理4通道图像"""
    train_transform = [
        albu.RandomRotate90(p=0.5),
    ]
    return albu.Compose(train_transform)

def get_normalize_only():
    """只归一化，用于处理RGB通道"""
    return albu.Compose([albu.Normalize()])


def train_aug(img, mask):
    crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                        SmartCropV1(crop_size=1024, max_ratio=0.75,
                                    ignore_index=len(CLASSES), nopad=False)])
    img, mask = crop_aug(img, mask)
    img, mask = np.array(img), np.array(mask)
    aug = get_training_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


def get_val_transform():
    val_transform = [
        albu.Normalize()
    ]
    return albu.Compose(val_transform)


def val_aug(img, mask):
    img, mask = np.array(img), np.array(mask)
    aug = get_val_transform()(image=img.copy(), mask=mask.copy())
    img, mask = aug['image'], aug['mask']
    return img, mask


class VaihingenDataset(Dataset):
    def __init__(self, data_root='data/vaihingen/test', mode='val', img_dir='images_1024', mask_dir='masks_1024',
                 img_suffix='.tif', mask_suffix='.png', transform=val_aug, mosaic_ratio=0.0,
                 img_size=ORIGIN_IMG_SIZE,
                 dsm_dir='dsm_1024', dsm_suffix='.tif', use_dsm=True):
        self.data_root = data_root
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transform = transform
        self.mode = mode
        self.mosaic_ratio = mosaic_ratio
        self.img_size = img_size
        self.dsm_dir = dsm_dir
        self.dsm_suffix = dsm_suffix
        self.use_dsm = use_dsm
        self.img_ids = self.get_img_ids(self.data_root, self.img_dir, self.mask_dir)

    def __getitem__(self, index):
        p_ratio = random.random()
        if p_ratio > self.mosaic_ratio or self.mode == 'val' or self.mode == 'test':
            img, mask, h_map = self.load_img_mask_and_dsm(index)
            if self.transform:
                img_np, mask_np, h_np = np.array(img), np.array(mask), None
                if h_map is not None:
                    h_np = np.array(h_map)
                    if h_np.ndim == 2:
                        h_np = h_np[..., None]
                    
                    # 保存DSM的原始值范围（用于恢复）
                    h_min, h_max = h_np.min(), h_np.max()
                    if h_max > h_min:
                        h_np_normalized = (h_np - h_min) / (h_max - h_min) * 255.0
                    else:
                        h_np_normalized = h_np.copy()
                    h_np_uint8 = h_np_normalized.astype(np.uint8)
                    
                    # 根据mode选择transform策略
                    if self.mode == 'train':
                        # 训练时：先做crop（需要同步应用到RGB和DSM），然后几何变换，最后归一化
                        # 1. Crop操作（同步应用到RGB和DSM）
                        crop_aug = Compose([RandomScale(scale_list=[0.5, 0.75, 1.0, 1.25, 1.5], mode='value'),
                                          SmartCropV1(crop_size=1024, max_ratio=0.75,
                                                      ignore_index=len(CLASSES), nopad=False)])
                        # 将RGB和DSM拼接，一起做crop
                        img_cat = np.concatenate([img_np, h_np_uint8], axis=2)
                        img_cat_pil = Image.fromarray(img_cat.astype(np.uint8))
                        mask_pil = Image.fromarray(mask_np)
                        img_cat_crop, mask_crop = crop_aug(img_cat_pil, mask_pil)
                        img_cat_crop = np.array(img_cat_crop)
                        mask_np = np.array(mask_crop)
                        
                        # 2. 几何变换（旋转）
                        geom_transform = get_training_transform_geom_only()
                        geom_result = geom_transform(image=img_cat_crop.copy(), mask=mask_np.copy())
                        img_cat_geom = geom_result['image']
                        mask_np = geom_result['mask']
                        
                        # 3. 分离RGB和DSM
                        img_np = img_cat_geom[:, :, :3]
                        h_np_uint8 = img_cat_geom[:, :, 3:]
                        
                        # 4. 恢复DSM的原始值范围
                        if h_max > h_min:
                            h_np = h_np_uint8.astype(np.float32) / 255.0 * (h_max - h_min) + h_min
                        else:
                            h_np = h_np_uint8.astype(np.float32)
                        if h_np.ndim == 3 and h_np.shape[2] == 1:
                            h_np = h_np[:, :, 0]
                        
                        # 5. 只对RGB归一化
                        normalize_transform = get_normalize_only()
                        norm_result = normalize_transform(image=img_np.copy(), mask=mask_np.copy())
                        img_np = norm_result['image']
                        mask_np = norm_result['mask']
                    else:
                        # 验证/测试时：只有归一化，直接分离后只对RGB归一化
                        normalize_transform = get_normalize_only()
                        
                        # 分离RGB和DSM（DSM不需要变换，保持原样）
                        img_np = img_np
                        # h_np保持原样，不需要变换
                        if h_np.ndim == 3 and h_np.shape[2] == 1:
                            h_np = h_np[:, :, 0]
                        
                        # 只对RGB归一化
                        norm_result = normalize_transform(image=img_np.copy(), mask=mask_np.copy())
                        img_np = norm_result['image']
                        mask_np = norm_result['mask']
                else:
                    # 没有DSM时，正常处理
                    img_np, mask_np = self.transform(img, mask)
                    img_np = np.array(img_np)
                img, mask = img_np, mask_np
                if h_np is not None:
                    h_map = h_np
        else:
            img, mask, h_map = self.load_mosaic_img_and_dsm(index)
            if self.transform:
                img, mask = self.transform(img, mask)

        img = torch.from_numpy(img).permute(2, 0, 1).float()
        mask = torch.from_numpy(mask).long()
        if h_map is not None:
            # 保证为 [1, H, W] 的浮点高度图
            if isinstance(h_map, Image.Image):
                h_map = np.array(h_map)
            if isinstance(h_map, np.ndarray):
                if h_map.ndim == 2:
                    h_map = h_map[None, ...]
                elif h_map.ndim == 3:
                    h_map = h_map.transpose(2, 0, 1)
                h_map = torch.from_numpy(h_map).float()
        else:
            h_map = torch.zeros(1, img.shape[1], img.shape[2], dtype=torch.float32)

        img_id = self.img_ids[index]
        results = dict(img_id=img_id, img=img, gt_semantic_seg=mask, h_map=h_map)
        return results

    def __len__(self):
        return len(self.img_ids)

    def get_img_ids(self, data_root, img_dir, mask_dir):
        img_filename_list = os.listdir(osp.join(data_root, img_dir))
        mask_filename_list = os.listdir(osp.join(data_root, mask_dir))
        assert len(img_filename_list) == len(mask_filename_list)
        img_ids = [str(id.split('.')[0]) for id in mask_filename_list]
        return img_ids

    def load_img_and_mask(self, index):
        img_id = self.img_ids[index]
        img_name = osp.join(self.data_root, self.img_dir, img_id + self.img_suffix)
        mask_name = osp.join(self.data_root, self.mask_dir, img_id + self.mask_suffix)
        img = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')
        return img, mask

    def load_img_mask_and_dsm(self, index):
        img, mask = self.load_img_and_mask(index)
        h_map = None
        if self.use_dsm and self.dsm_dir is not None:
            img_id = self.img_ids[index]
            # DSM patch 命名规则：与 RGB patch 基名一致（例如：top_mosaic_09cm_area1_0_0.tif）
            dsm_name = osp.join(self.data_root, self.dsm_dir, img_id + self.dsm_suffix)
            if osp.exists(dsm_name):
                h_map = Image.open(dsm_name)
            else:
                h_map = None
        return img, mask, h_map

    def load_mosaic_img_and_mask(self, index):
        indexes = [index] + [random.randint(0, len(self.img_ids) - 1) for _ in range(3)]
        img_a, mask_a = self.load_img_and_mask(indexes[0])
        img_b, mask_b = self.load_img_and_mask(indexes[1])
        img_c, mask_c = self.load_img_and_mask(indexes[2])
        img_d, mask_d = self.load_img_and_mask(indexes[3])

        img_a, mask_a = np.array(img_a), np.array(mask_a)
        img_b, mask_b = np.array(img_b), np.array(mask_b)
        img_c, mask_c = np.array(img_c), np.array(mask_c)
        img_d, mask_d = np.array(img_d), np.array(mask_d)

        h = self.img_size[0]
        w = self.img_size[1]

        start_x = w // 4
        strat_y = h // 4
        # The coordinates of the splice center
        offset_x = random.randint(start_x, (w - start_x))
        offset_y = random.randint(strat_y, (h - strat_y))

        crop_size_a = (offset_x, offset_y)
        crop_size_b = (w - offset_x, offset_y)
        crop_size_c = (offset_x, h - offset_y)
        crop_size_d = (w - offset_x, h - offset_y)

        random_crop_a = albu.RandomCrop(width=crop_size_a[0], height=crop_size_a[1])
        random_crop_b = albu.RandomCrop(width=crop_size_b[0], height=crop_size_b[1])
        random_crop_c = albu.RandomCrop(width=crop_size_c[0], height=crop_size_c[1])
        random_crop_d = albu.RandomCrop(width=crop_size_d[0], height=crop_size_d[1])

        croped_a = random_crop_a(image=img_a.copy(), mask=mask_a.copy())
        croped_b = random_crop_b(image=img_b.copy(), mask=mask_b.copy())
        croped_c = random_crop_c(image=img_c.copy(), mask=mask_c.copy())
        croped_d = random_crop_d(image=img_d.copy(), mask=mask_d.copy())

        img_crop_a, mask_crop_a = croped_a['image'], croped_a['mask']
        img_crop_b, mask_crop_b = croped_b['image'], croped_b['mask']
        img_crop_c, mask_crop_c = croped_c['image'], croped_c['mask']
        img_crop_d, mask_crop_d = croped_d['image'], croped_d['mask']

        top = np.concatenate((img_crop_a, img_crop_b), axis=1)
        bottom = np.concatenate((img_crop_c, img_crop_d), axis=1)
        img = np.concatenate((top, bottom), axis=0)

        top_mask = np.concatenate((mask_crop_a, mask_crop_b), axis=1)
        bottom_mask = np.concatenate((mask_crop_c, mask_crop_d), axis=1)
        mask = np.concatenate((top_mask, bottom_mask), axis=0)
        mask = np.ascontiguousarray(mask)
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        mask = Image.fromarray(mask)
        # print(img.shape)

        return img, mask

    def load_mosaic_img_and_dsm(self, index):
        # 目前 mosaic 仅对 RGB 和 mask 进行拼接，DSM 直接退化为 None
        img, mask = self.load_mosaic_img_and_mask(index)
        h_map = None
        return img, mask, h_map


def show_img_mask_seg(seg_path, img_path, mask_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 3, figsize=(18, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        mask = cv2.imread(f'{mask_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        mask = mask.astype(np.uint8)
        mask = Image.fromarray(mask).convert('P')
        mask.putpalette(np.array(PALETTE, dtype=np.uint8))
        mask = np.array(mask.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE ' + img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(mask)
        ax[i, 1].set_title('Mask True ' + seg_id)
        ax[i, 2].set_axis_off()
        ax[i, 2].imshow(img_seg)
        ax[i, 2].set_title('Mask Predict ' + seg_id)
        ax[i, 2].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_seg(seg_path, img_path, start_seg_index):
    seg_list = os.listdir(seg_path)
    seg_list = [f for f in seg_list if f.endswith('.png')]
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    seg_list = seg_list[start_seg_index:start_seg_index+2]
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    for i in range(len(seg_list)):
        seg_id = seg_list[i]
        img_seg = cv2.imread(f'{seg_path}/{seg_id}', cv2.IMREAD_UNCHANGED)
        img_seg = img_seg.astype(np.uint8)
        img_seg = Image.fromarray(img_seg).convert('P')
        img_seg.putpalette(np.array(PALETTE, dtype=np.uint8))
        img_seg = np.array(img_seg.convert('RGB'))
        img_id = str(seg_id.split('.')[0])+'.tif'
        img = cv2.imread(f'{img_path}/{img_id}', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax[i, 0].set_axis_off()
        ax[i, 0].imshow(img)
        ax[i, 0].set_title('RS IMAGE '+img_id)
        ax[i, 1].set_axis_off()
        ax[i, 1].imshow(img_seg)
        ax[i, 1].set_title('Seg IMAGE '+seg_id)
        ax[i, 1].legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')


def show_mask(img, mask, img_id):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 12))
    patches = [mpatches.Patch(color=np.array(PALETTE[i])/255., label=CLASSES[i]) for i in range(len(CLASSES))]
    mask = mask.astype(np.uint8)
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(PALETTE, dtype=np.uint8))
    mask = np.array(mask.convert('RGB'))
    ax1.imshow(img)
    ax1.set_title('RS IMAGE ' + str(img_id)+'.tif')
    ax2.imshow(mask)
    ax2.set_title('Mask ' + str(img_id)+'.png')
    ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
