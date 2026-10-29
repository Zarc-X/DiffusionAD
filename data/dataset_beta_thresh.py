import os
import numpy as np
from torch.utils.data import Dataset
import torch
import cv2
import glob
import imgaug.augmenters as iaa
from PIL import Image
from torchvision import transforms
import random
from data.perlin import rand_perlin_2d_np

texture_list = ['carpet', 'zipper', 'leather', 'tile', 'wood','grid',
                'Class1', 'Class2', 'Class3', 'Class4', 'Class5',
                 'Class6', 'Class7', 'Class8', 'Class9', 'Class10']


LINE_DEFECT_CLASSES = {"screw", "transistor", "toothbrush", "cable"}


def _cfg_float(args, key, default):
    value = args.get(key, default)
    if value is None or value == "":
        return float(default)
    try:
        return float(value)
    except Exception:
        return float(default)


def _cfg_int(args, key, default):
    value = args.get(key, default)
    if value is None or value == "":
        return int(default)
    try:
        return int(value)
    except Exception:
        return int(default)


def _cfg_bool(args, key, default):
    value = args.get(key, default)
    if value is None or value == "":
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _clip01(x):
    return np.clip(x, 0.0, 1.0)


def _init_synth_cfg(args, classname):
    cfg = {
        "mode": str(args.get("anomaly_synth_mode", "legacy")).strip().lower(),
        "no_anomaly_prob": _cfg_float(args, "anomaly_no_anomaly_prob", 0.5),
        "perlin_min_scale": _cfg_int(args, "anomaly_perlin_min_scale", 0),
        "perlin_max_scale": _cfg_int(args, "anomaly_perlin_max_scale", 6),
        "perlin_threshold": _cfg_float(args, "anomaly_perlin_threshold", 0.5),
        "mask_try_count": _cfg_int(args, "anomaly_mask_try_count", 50),
        "multi_scale_count": _cfg_int(args, "anomaly_multi_scale_count", 2),
        "target_area_min": _cfg_float(args, "anomaly_target_area_min", 0.01),
        "target_area_max": _cfg_float(args, "anomaly_target_area_max", 0.12),
        "texture_mix_prob": _cfg_float(args, "anomaly_texture_mix_prob", 0.5),
        "alpha_min": _cfg_float(args, "anomaly_alpha_min", 0.35),
        "alpha_max": _cfg_float(args, "anomaly_alpha_max", 0.8),
        "feather_ksize": _cfg_int(args, "anomaly_feather_ksize", 7),
        "color_transfer": _cfg_bool(args, "anomaly_color_transfer", True),
        "line_defect_prob": _cfg_float(args, "anomaly_line_defect_prob", 0.45),
        "line_count_min": _cfg_int(args, "anomaly_line_count_min", 1),
        "line_count_max": _cfg_int(args, "anomaly_line_count_max", 3),
        "line_thickness_min": _cfg_int(args, "anomaly_line_thickness_min", 1),
        "line_thickness_max": _cfg_int(args, "anomaly_line_thickness_max", 3),
        "line_length_min": _cfg_float(args, "anomaly_line_length_min", 0.12),
        "line_length_max": _cfg_float(args, "anomaly_line_length_max", 0.45),
    }

    cfg["no_anomaly_prob"] = float(np.clip(cfg["no_anomaly_prob"], 0.0, 0.95))
    cfg["perlin_min_scale"] = max(0, int(cfg["perlin_min_scale"]))
    cfg["perlin_max_scale"] = max(cfg["perlin_min_scale"] + 1, int(cfg["perlin_max_scale"]))
    cfg["perlin_threshold"] = float(np.clip(cfg["perlin_threshold"], 0.05, 0.95))
    cfg["mask_try_count"] = max(1, int(cfg["mask_try_count"]))
    cfg["multi_scale_count"] = max(1, int(cfg["multi_scale_count"]))
    cfg["target_area_min"] = float(np.clip(cfg["target_area_min"], 0.001, 0.35))
    cfg["target_area_max"] = float(np.clip(cfg["target_area_max"], cfg["target_area_min"] + 1e-4, 0.45))
    cfg["texture_mix_prob"] = float(np.clip(cfg["texture_mix_prob"], 0.0, 1.0))
    cfg["alpha_min"] = float(np.clip(cfg["alpha_min"], 0.05, 0.95))
    cfg["alpha_max"] = float(np.clip(cfg["alpha_max"], cfg["alpha_min"], 0.98))
    cfg["feather_ksize"] = max(1, int(cfg["feather_ksize"]))
    if cfg["feather_ksize"] % 2 == 0:
        cfg["feather_ksize"] += 1
    cfg["line_defect_prob"] = float(np.clip(cfg["line_defect_prob"], 0.0, 1.0))
    cfg["line_count_min"] = max(1, int(cfg["line_count_min"]))
    cfg["line_count_max"] = max(cfg["line_count_min"], int(cfg["line_count_max"]))
    cfg["line_thickness_min"] = max(1, int(cfg["line_thickness_min"]))
    cfg["line_thickness_max"] = max(cfg["line_thickness_min"], int(cfg["line_thickness_max"]))
    cfg["line_length_min"] = float(np.clip(cfg["line_length_min"], 0.02, 0.95))
    cfg["line_length_max"] = float(np.clip(cfg["line_length_max"], cfg["line_length_min"], 1.2))

    cls = str(classname).strip().lower()
    if cls in LINE_DEFECT_CLASSES:
        cfg["target_area_min"] = min(cfg["target_area_min"], 0.01)
        cfg["target_area_max"] = min(cfg["target_area_max"], 0.06)
    return cfg


def _build_patch_shuffle(base_img):
    high, width = base_img.shape[0], base_img.shape[1]
    grid_h = max(2, int(high / 8))
    grid_w = max(2, int(width / 8))

    wi = np.split(base_img, range(grid_w, width, grid_w), axis=1)
    wi1 = wi[::2]
    random.shuffle(wi1)
    wi2 = wi[1::2]
    random.shuffle(wi2)
    width_cut_image = np.concatenate((np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)

    hi = np.split(width_cut_image, range(grid_h, high, grid_h), axis=0)
    random.shuffle(hi)
    hi1 = hi[::2]
    random.shuffle(hi1)
    hi2 = hi[1::2]
    random.shuffle(hi2)
    return np.concatenate((np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)


def _color_transfer(source, target, mask_2d):
    mask = mask_2d > 0.5
    if int(mask.sum()) < 16:
        return source

    src_pixels = source[mask]
    tgt_pixels = target[mask]
    src_mean = src_pixels.mean(axis=0)
    src_std = src_pixels.std(axis=0) + 1e-6
    tgt_mean = tgt_pixels.mean(axis=0)
    tgt_std = tgt_pixels.std(axis=0) + 1e-6

    transferred = (source - src_mean) / src_std * tgt_std + tgt_mean
    return _clip01(transferred)


def _draw_line_mask(valid_mask, cfg, classname):
    h, w = valid_mask.shape
    coords = np.argwhere(valid_mask > 0)
    if coords.shape[0] == 0:
        return np.zeros((h, w), dtype=np.uint8)

    cls = str(classname).strip().lower()
    line_cnt = np.random.randint(cfg["line_count_min"], cfg["line_count_max"] + 1)
    line_mask = np.zeros((h, w), dtype=np.uint8)
    min_hw = float(min(h, w))

    for _ in range(line_cnt):
        yi, xi = coords[np.random.randint(0, coords.shape[0])]
        theta = np.random.uniform(0.0, np.pi)
        if cls == "transistor":
            length_ratio = np.random.uniform(0.06, min(0.30, cfg["line_length_max"]))
        else:
            length_ratio = np.random.uniform(cfg["line_length_min"], cfg["line_length_max"])
        length = max(4, int(length_ratio * min_hw))

        x2 = int(np.clip(xi + length * np.cos(theta), 0, w - 1))
        y2 = int(np.clip(yi + length * np.sin(theta), 0, h - 1))
        thickness = np.random.randint(cfg["line_thickness_min"], cfg["line_thickness_max"] + 1)
        cv2.line(line_mask, (int(xi), int(yi)), (x2, y2), color=1, thickness=thickness)

    line_mask = line_mask * valid_mask.astype(np.uint8)
    return line_mask


def _build_realistic_mask(dataset, thresh):
    cfg = dataset.synth_cfg
    h, w = dataset.resize_shape[0], dataset.resize_shape[1]
    valid_mask = (thresh > 0.5).astype(np.uint8)
    if int(valid_mask.sum()) < 16:
        valid_mask = np.ones((h, w), dtype=np.uint8)

    accum = np.zeros((h, w), dtype=np.float32)
    for _ in range(cfg["multi_scale_count"]):
        sx = 2 ** np.random.randint(cfg["perlin_min_scale"], cfg["perlin_max_scale"])
        sy = 2 ** np.random.randint(cfg["perlin_min_scale"], cfg["perlin_max_scale"])
        noise = rand_perlin_2d_np((h, w), (sx, sy))
        noise = dataset.rot(image=noise)
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
        accum += noise * np.random.uniform(0.7, 1.3)

    accum = accum / float(cfg["multi_scale_count"])
    cls = str(dataset.classname).strip().lower()

    area_min = cfg["target_area_min"]
    area_max = cfg["target_area_max"]
    if cls in {"wood", "carpet", "leather"}:
        area_min = float(np.clip(area_min * 1.15, 0.001, 0.35))
        area_max = float(np.clip(area_max * 1.20, area_min + 1e-4, 0.45))

    valid_vals = accum[valid_mask > 0]
    if valid_vals.size < 32:
        return np.zeros((h, w), dtype=np.float32)

    target_ratio = np.random.uniform(area_min, area_max)
    q = float(np.clip(1.0 - target_ratio, 0.0, 1.0))
    threshold = np.quantile(valid_vals, q)
    mask = ((accum >= threshold) & (valid_mask > 0)).astype(np.uint8)

    if int(mask.sum()) == 0:
        mask = ((accum >= cfg["perlin_threshold"]) & (valid_mask > 0)).astype(np.uint8)

    kernel_radius = np.random.randint(1, 4)
    kernel = np.ones((2 * kernel_radius + 1, 2 * kernel_radius + 1), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    line_prob = cfg["line_defect_prob"]
    if np.random.rand() < line_prob:
        line_mask = _draw_line_mask(valid_mask, cfg, dataset.classname)
        mask = np.maximum(mask, line_mask)

    mask = (mask * valid_mask).astype(np.float32)
    return mask


def _build_anomaly_texture(dataset, image, anomaly_source_path, cv2_image, object_mask):
    cfg = dataset.synth_cfg
    h, w = dataset.resize_shape[0], dataset.resize_shape[1]
    cls = str(dataset.classname).strip().lower()

    force_texture = cls in texture_list
    use_texture = force_texture or (np.random.rand() < cfg["texture_mix_prob"])

    texture = None
    if use_texture and anomaly_source_path and os.path.exists(anomaly_source_path):
        src = cv2.imread(anomaly_source_path)
        if src is not None:
            src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
            src = cv2.resize(src, dsize=(w, h))
            texture = dataset.randAugmenter()(image=src)

    if texture is None:
        base = cv2_image
        if base.dtype != np.uint8:
            base = (np.clip(base, 0, 255)).astype(np.uint8)
        aug = dataset.randAugmenter()(image=base)
        texture = _build_patch_shuffle(aug)

    texture = texture.astype(np.float32) / 255.0
    if cfg["color_transfer"]:
        texture = _color_transfer(texture, image, object_mask)

    return _clip01(texture)


def _realistic_synthetic(dataset, image, thresh, anomaly_source_path, cv2_image):
    cfg = dataset.synth_cfg
    no_anomaly = np.random.rand()
    if no_anomaly < cfg["no_anomaly_prob"]:
        return image.astype(np.float32), np.zeros((dataset.resize_shape[0], dataset.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

    mask_2d = np.zeros((dataset.resize_shape[0], dataset.resize_shape[1]), dtype=np.float32)
    for _ in range(cfg["mask_try_count"]):
        mask_2d = _build_realistic_mask(dataset, thresh)
        if float(mask_2d.sum()) > 0:
            break

    if float(mask_2d.sum()) <= 0:
        return image.astype(np.float32), np.zeros((dataset.resize_shape[0], dataset.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

    anomaly_tex = _build_anomaly_texture(dataset, image, anomaly_source_path, cv2_image, mask_2d)

    feather_ksize = cfg["feather_ksize"]
    if feather_ksize > 1:
        soft_mask = cv2.GaussianBlur(mask_2d, (feather_ksize, feather_ksize), 0)
    else:
        soft_mask = mask_2d
    soft_mask = _clip01(soft_mask)

    alpha = np.random.uniform(cfg["alpha_min"], cfg["alpha_max"])
    alpha_mask = np.expand_dims(alpha * soft_mask, axis=2)

    augmented = image * (1.0 - alpha_mask) + anomaly_tex * alpha_mask

    augmented = _clip01(augmented).astype(np.float32)
    msk = np.expand_dims((mask_2d > 0).astype(np.float32), axis=2)
    return augmented, msk, np.array([1.0], dtype=np.float32)
class MVTecTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class MVTecTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):

        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.synth_cfg = _init_synth_cfg(args, self.classname)
        

        #foreground path of textural classes
        foreground_path = os.path.join(args["mvtec_root_path"],'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

        

    
    def __len__(self):
        return len(self.image_paths)

    def random_choice_foreground_path(self):
        foreground_path_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_path_id]
        return foreground_path


    def get_foreground_mvtec(self,image_path):
        classname = self.classname
        if classname in texture_list:
            foreground_path = self.random_choice_foreground_path()
        else:
            foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path



    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):
        if self.synth_cfg["mode"] == "realistic":
            return _realistic_synthetic(self, image, thresh, anomaly_source_path, cv2_image)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  
                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32)  

                msk = (object_perlin).astype(np.float32) 
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
                
            
            if self.classname in texture_list: # only DTD
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else: # DTD and self-augmentation
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  # >0.5 is DTD 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else: #self-augmentation
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2
            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground_mvtec(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0 
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample



class VisATestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.JPG"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+".png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class VisATrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]
        self.image_paths = sorted(glob.glob(self.root_dir+"/*.JPG"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))
        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.synth_cfg = _init_synth_cfg(args, self.classname)
    
    def __len__(self):
        return len(self.image_paths)


    def get_foreground(self,image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path 


    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):
        if self.synth_cfg["mode"] == "realistic":
            return _realistic_synthetic(self, image, thresh, anomaly_source_path, cv2_image)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  

            min_perlin_scale = 0


            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50):  

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(
                    np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0

            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class DAGMTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.PNG"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_label.PNG"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class DAGMTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.PNG"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.synth_cfg = _init_synth_cfg(args, self.classname)
        
        #foreground path of textural classes
        foreground_path = os.path.join(args["mvtec_root_path"],'carpet')
        self.textural_foreground_path = sorted(glob.glob(foreground_path +"/thresh/*.png"))

        

    
    def __len__(self):
        return len(self.image_paths)

    def random_choice_foreground(self):
        foreground_id = torch.randint(0, len(self.textural_foreground_path), (1,)).item()
        foreground_path = self.textural_foreground_path[foreground_id]
        return foreground_path

    def get_foreground(self,image_path):
        foreground_path = self.random_choice_foreground()
        return foreground_path 

    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):
        if self.synth_cfg["mode"] == "realistic":
            return _realistic_synthetic(self, image, thresh, anomaly_source_path, cv2_image)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6
            min_perlin_scale = 0


            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50): 

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(
                    np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0:
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5:  
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate(
                        (np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)

                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0


            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  
        image = np.array(image).astype(np.float32)/255.0

        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample


class MPDDTestDataset(Dataset):

    def __init__(self, data_path,classname,img_size):
        self.root_dir = os.path.join(data_path,'test')
        self.images = sorted(glob.glob(self.root_dir+"/*/*.png"))
        self.resize_shape = [img_size[0], img_size[1]]

    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        if mask_path is not None:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((image.shape[0], image.shape[1]))
        if self.resize_shape != None:
            image = cv2.resize(image, dsize=(
                self.resize_shape[1], self.resize_shape[0]))
            mask = cv2.resize(mask, dsize=(
                self.resize_shape[1], self.resize_shape[0]))

        image = image / 255.0
        mask = mask / 255.0

        image = np.array(image).reshape(
            (image.shape[0], image.shape[1], 3)).astype(np.float32)
        mask = np.array(mask).reshape(
            (mask.shape[0], mask.shape[1], 1)).astype(np.float32)

        image = np.transpose(image, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))
        return image, mask
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)

        sample = {'image': image, 'has_anomaly': has_anomaly,
                  'mask': mask, 'idx': idx,'type':img_path[len(self.root_dir):-8],'file_name':base_dir+'_'+file_name}

        return sample

class MPDDTrainDataset(Dataset):

    def __init__(self, data_path,classname,img_size,args):
        self.classname=classname
        self.root_dir = os.path.join(data_path,'train','good')
        self.resize_shape = [img_size[0], img_size[1]]
        self.anomaly_source_path = args["anomaly_source_path"]

        self.image_paths = sorted(glob.glob(self.root_dir+"/*.png"))
        self.anomaly_source_paths = sorted(glob.glob(self.anomaly_source_path+"/images/*/*.jpg"))

        self.augmenters = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           iaa.Affine(rotate=(-45, 45))
                           ]
        
        self.augmenters_anomaly = [iaa.GammaContrast((0.5, 2.0), per_channel=True),
                           iaa.MultiplyAndAddToBrightness(
                               mul=(0.8, 1.2), add=(-30, 30)),
                           iaa.pillike.EnhanceSharpness(),
                           iaa.AddToHueAndSaturation(
                               (-50, 50), per_channel=True),
                           iaa.Solarize(0.5, threshold=(32, 128)),
                           iaa.Posterize(),
                           iaa.Invert(),
                           iaa.pillike.Autocontrast(),
                           iaa.pillike.Equalize(),
                           ]

        self.augmenters_mask = [iaa.Affine(rotate=(-90, 90)),
                              iaa.Affine(shear=(0, 40)),
                           iaa.Affine(translate_percent={"x": (-0.5, 0.5), "y": (-0.5, 0.5)}),]
        
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])
        self.synth_cfg = _init_synth_cfg(args, self.classname)
        
        

    
    def __len__(self):
        return len(self.image_paths)

    def get_foreground(self,image_path):
        foreground_path = image_path.replace('train', 'DISthresh')
        return foreground_path 


    def randAugmenter_anomaly(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_anomaly)), 2, replace=False)
        aug = iaa.Sequential([self.augmenters_anomaly[aug_ind[0]],
                              self.augmenters_anomaly[aug_ind[1]]]
                             )
        return aug

    def randAugmenter_mask(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters_mask)), 1, replace=False)
        aug = iaa.Sequential([self.augmenters_mask[aug_ind[0]],]
                             )
        return aug


    def randAugmenter(self):
        aug_ind = np.random.choice(
            np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug


    def perlin_synthetic(self, image, thresh, anomaly_source_path, cv2_image,thresh_path):
        if self.synth_cfg["mode"] == "realistic":
            return _realistic_synthetic(self, image, thresh, anomaly_source_path, cv2_image)

        no_anomaly = torch.rand(1).numpy()[0]
        if no_anomaly > 0.5:
            image = image.astype(np.float32)
            return image, np.zeros((self.resize_shape[0], self.resize_shape[1], 1), dtype=np.float32), np.array([0.0], dtype=np.float32)

        else:
            perlin_scale = 6  
            min_perlin_scale = 0
            perlin_scalex = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])
            perlin_scaley = 2 ** (torch.randint(min_perlin_scale,
                                  perlin_scale, (1,)).numpy()[0])

            has_anomaly = 0
            try_cnt = 0
            while(has_anomaly == 0 and try_cnt<50): 

                perlin_noise = rand_perlin_2d_np(
                    (self.resize_shape[0], self.resize_shape[1]), (perlin_scalex, perlin_scaley))
                perlin_noise = self.rot(image=perlin_noise)
                threshold = 0.5
                perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
                
                object_perlin = thresh*perlin_thr

                object_perlin = np.expand_dims(object_perlin, axis=2).astype(np.float32) 

                msk = (object_perlin).astype(np.float32)  
                if np.sum(msk) !=0: 
                    has_anomaly = 1        
                try_cnt+=1
            if self.classname in texture_list:
                aug = self.randAugmenter()
                anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                    self.resize_shape[1], self.resize_shape[0]))
                anomaly_img_augmented = aug(image=anomaly_source_img)
                img_object_thr = anomaly_img_augmented.astype(
                    np.float32) * object_perlin/255.0
            else:
                texture_or_patch = torch.rand(1).numpy()[0]
                if texture_or_patch > 0.5: 
                    aug = self.randAugmenter()
                    anomaly_source_img = cv2.cvtColor(cv2.imread(anomaly_source_path),cv2.COLOR_BGR2RGB)
                    anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(
                        self.resize_shape[1], self.resize_shape[0]))
                    anomaly_img_augmented = aug(image=anomaly_source_img)
                    img_object_thr = anomaly_img_augmented.astype(
                        np.float32) * object_perlin/255.0

                else:
                    aug = self.randAugmenter()
                    anomaly_image = aug(image=cv2_image)
                    high, width = anomaly_image.shape[0], anomaly_image.shape[1]
                    gird_high, gird_width = int(high/8), int(width/8)
                    wi = np.split(anomaly_image, range(
                        gird_width, width, gird_width), axis=1)
                    wi1 = wi[::2]
                    random.shuffle(wi1)
                    wi2 = wi[1::2]
                    random.shuffle(wi2)
                    width_cut_image = np.concatenate((np.concatenate(wi1, axis=1), np.concatenate(wi2, axis=1)), axis=1)
                    hi = np.split(width_cut_image, range(
                        gird_high, high, gird_high), axis=0)
                    random.shuffle(hi)
                    hi1 = hi[::2]
                    random.shuffle(hi1)
                    hi2 = hi[1::2]
                    random.shuffle(hi2)
                    mixer_cut_image = np.concatenate(
                        (np.concatenate(hi1, axis=0), np.concatenate(hi2, axis=0)), axis=0)
                    img_object_thr = mixer_cut_image.astype(
                        np.float32) * object_perlin/255.0


            beta = torch.rand(1).numpy()[0] * 0.6 + 0.2

            augmented_image = image * \
                (1 - object_perlin) + (1 - beta) * \
                img_object_thr + beta * image * (object_perlin)

            augmented_image = augmented_image.astype(np.float32)

            return augmented_image, msk, np.array([has_anomaly], dtype=np.float32)


    def __getitem__(self, idx):
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        image_path = self.image_paths[idx]
        image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        cv2_image=image
        thresh_path = self.get_foreground(image_path)
        thresh=cv2.imread(thresh_path,0)
        thresh = cv2.resize(thresh,dsize=(self.resize_shape[1], self.resize_shape[0]))

        thresh = np.array(thresh).astype(np.float32)/255.0  # [0,255]->[0,1]
        image = np.array(image).astype(np.float32)/255.0


        
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        anomaly_path = self.anomaly_source_paths[anomaly_source_idx]
        augmented_image, anomaly_mask, has_anomaly  = self.perlin_synthetic(image,thresh,anomaly_path,cv2_image,thresh_path)
        
        augmented_image = np.transpose(augmented_image, (2, 0, 1))
        image = np.transpose(image, (2, 0, 1))
        anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))


        sample = {'image': image, "anomaly_mask": anomaly_mask,
                  'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}

        return sample

