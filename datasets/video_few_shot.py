import os
import random
import torch
import torch.utils.data
import utils.logging as logging
import numpy as np
import traceback
from torchvision.transforms import Compose
import torchvision.transforms._transforms_video as transforms
from datasets.augmentations import (
    ColorJitter,
    KineticsResizedCropFewshot,
    RandomErasing
)
from datasets.base_dataset import BaseVideoDataset
from datasets.builder import DATASET_REGISTRY

logger = logging.get_logger(__name__)

class Split_few_shot():
    """Contains video frame paths and ground truth labels for a single split (e.g. train videos). """
    def __init__(self, folder, split_dataset='train', dataset="Video_few_shot"):
        self.gt_a_list = []
        self.videos = []
        self.split_dataset = split_dataset
        if dataset == 'Video_few_shot':
            for class_folder in folder:
                paths = class_folder.strip().split('/')[-1]
                class_id = int(class_folder.strip().split('/')[0][len(split_dataset):]) # class_folders.index(class_folder)
                self.add_vid(paths, class_id)
        else:
            for class_folder in folder:
                paths = class_folder.strip().split('//')[-1]
                class_id = int(class_folder.strip().split('//')[0][len(split_dataset):]) # class_folders.index(class_folder)
                self.add_vid(paths, class_id)
        logger.info("loaded {} videos from {} dataset: {} !".format(len(self.gt_a_list), split_dataset, dataset))

    def add_vid(self, paths, gt_a):
        self.videos.append(paths)
        self.gt_a_list.append(gt_a)

    def get_rand_vid(self, label, idx=-1):
        match_idxs = []
        for i in range(len(self.gt_a_list)):
            if label == self.gt_a_list[i]:
                match_idxs.append(i)
        if idx != -1:
            return self.videos[match_idxs[idx]], match_idxs[idx]
        random_idx = np.random.choice(match_idxs)
        return self.videos[random_idx], random_idx

    def get_single_video(self, index):
        return self.videos[index], self.gt_a_list[index]

    def get_num_videos_for_class(self, label):
        return len([gt for gt in self.gt_a_list if gt == label])

    def get_unique_classes(self):
        return list(set(self.gt_a_list))

    def __len__(self):
        return len(self.gt_a_list)

@DATASET_REGISTRY.register()
class Video_few_shot(BaseVideoDataset):
    def __init__(self, cfg, split):
        super(Video_few_shot, self).__init__(cfg, split) 
        if self.split == "test":
            self._pre_transformation_config_required = True
        self.split_dataset = split     
        
    def _get_dataset_list_name(self):
        """
            Returns:
                dataset_list_name (string)
        """
        name = "{}_few_shot.txt".format(self.split)
        logger.info("Reading video list from file: {}".format(name))
        return name

    def _get_sample_info(self, index):
        """
            Input: 
                index (int): video index
            Returns:
                sample_info (dict): contains different informations to be used later
                    Things that must be included are:
                    "video_path" indicating the video's path w.r.t. index
                    "supervised_label" indicating the class of the video 
        """
        class_ = self._samples[index]["label_idx"]
        video_path = os.path.join(self.data_root_dir, self._samples[index]["id"]+".mp4")
        sample_info = {
            "path": video_path,
            "supervised_label": class_,
        }
        return sample_info
    
    def _construct_dataset(self, cfg):
        if hasattr(self.cfg.TRAIN, "DATASET_FEW"):
            self.dataset_name = self.cfg.TRAIN.DATASET_FEW
        self._num_clips = 1
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()
        for retry in range(5):
            try:
                logger.info("Loading {} dataset list for split '{}'...".format(self.dataset_name, self.split))
                local_file = os.path.join(cfg.OUTPUT_DIR, dataset_list_name)
                if not os.path.exists(local_file):
                    local_file = os.path.join(self.anno_dir, dataset_list_name)
                assert os.path.exists(local_file), f"Data list {local_file} not found."
                if local_file[-4:] == ".csv":
                    import pandas
                    lines = pandas.read_csv(local_file)
                    for line in lines.values.tolist():
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                elif local_file[-4:] == "json":
                    import json
                    with open(local_file, "r") as f:
                        lines = json.load(f)
                    for line in lines:
                        for idx in range(self._num_clips):
                            self._samples.append(line)
                            self._spatial_temporal_index.append(idx)
                else:
                    with open(local_file) as f:
                        lines = f.readlines()
                        for line in lines:
                            for idx in range(self._num_clips):
                                self._samples.append(line.strip())
                                self._spatial_temporal_index.append(idx)
                self.split_few_shot = Split_few_shot(lines, self.split, dataset=self.dataset_name)
                logger.info("Dataset {} split {} loaded. Length {}.".format(self.dataset_name, self.split, len(self._samples)))
                break
            except:
                if retry<4:
                    continue
                else:
                    raise ValueError("Data list {} not found.".format(os.path.join(self.anno_dir, dataset_list_name)))

    def __getitem__(self, index):
        """
            Returns:
                frames (dict): {
                    "video": (tensor), 
                    "text_embedding" (optional): (tensor)
                }
                labels (dict): {
                    "supervised": (tensor),
                    "self-supervised" (optional): (...)
                }
        """
        if self.cfg.TRAIN.META_BATCH:
            """returns dict of support and target images and labels for a meta training task"""
            #select classes to use for this task
            c = self.split_few_shot
            classes = c.get_unique_classes()
            batch_classes = random.sample(classes, self.cfg.TRAIN.WAY)
            if self.split != "train" and hasattr(self.cfg.TRAIN, "WAT_TEST"):
                batch_classes = random.sample(classes, self.cfg.TRAIN.WAT_TEST)
            if self.split == "train":
                n_queries = self.cfg.TRAIN.QUERY_PER_CLASS
                SHOT = self.cfg.TRAIN.SHOT
            else:
                n_queries = self.cfg.TRAIN.QUERY_PER_CLASS_TEST
                if hasattr(self.cfg.TRAIN, "SHOT_TEST"):
                    SHOT = self.cfg.TRAIN.SHOT_TEST
                else:
                    SHOT = self.cfg.TRAIN.SHOT
            retries = 5
            for retry in range(retries):
                try:
                    support_set = []
                    support_labels = []
                    target_set = []
                    target_labels = []
                    real_support_labels = []
                    real_target_labels = []
                    for bl, bc in enumerate(batch_classes):
                        n_total = c.get_num_videos_for_class(bc)
                        idxs = random.sample([i for i in range(n_total)], SHOT + n_queries)
                        for idx in idxs[0:SHOT]:
                            if hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF_SUPPORT") and self.cfg.AUGMENTATION.SUPPORT_QUERY_DIFF_SUPPORT and self.split_dataset=="train":
                                vid, _ = self.get_seq_query(bc, idx)
                            else:
                                vid, _ = self.get_seq(bc, idx)
                            support_set.append(vid)
                            support_labels.append(bl)
                            real_support_labels.append(bc)
                        # try:
                        for idx in idxs[SHOT:]:
                            if hasattr(self.cfg.AUGMENTATION, "SUPPORT_QUERY_DIFF") and self.cfg.AUGMENTATION.SUPPORT_QUERY_DIFF and self.split_dataset=="train":
                                vid, _ = self.get_seq_query(bc, idx)
                            else:
                                vid, _ = self.get_seq(bc, idx)
                            target_set.append(vid)
                            target_labels.append(bl)
                            real_target_labels.append(bc)
                    break   
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, idxs, bc
                    ))
            s = list(zip(support_set, support_labels, real_support_labels))
            random.shuffle(s)
            support_set, support_labels, real_support_labels = zip(*s)
            t = list(zip(target_set, target_labels, real_target_labels))
            random.shuffle(t)
            target_set, target_labels, real_target_labels = zip(*t)
            support_set = torch.cat(support_set)  
            target_set = torch.cat(target_set)    
            support_labels = torch.FloatTensor(support_labels)
            target_labels = torch.FloatTensor(target_labels)
            real_target_labels = torch.FloatTensor(real_target_labels)  
            real_support_labels = torch.FloatTensor(real_support_labels)
            batch_classes = torch.FloatTensor(batch_classes)
            return {"support_set":support_set, "support_labels":support_labels, "target_set":target_set, "target_labels":target_labels, "real_target_labels":real_target_labels, "batch_class_list": batch_classes, "real_support_labels":real_support_labels}
        else:
            sample_info = self._get_sample_info(index)
            retries = 1 if self.split == "train" else 10
            for retry in range(retries):
                try:
                    data, _, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, index, sample_info["path"]
                    ))
            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if self.split in ["test"] and self.cfg.TEST.ZERO_SHOT:
                if not hasattr(self, "label_embd"):
                    self.label_embd = self.word_embd(self.words_to_ids(self.label_names))
                data["text_embedding"] = self.label_embd
            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()
            meta = {}
            return data, [], index, meta

    def get_seq(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.split_few_shot
        if self.cfg.TRAIN.META_BATCH:
            paths, vid_id = c.get_rand_vid(label, idx) 
            if self.dataset_name == 'Video_few_shot':
                video_path = os.path.join(self.data_root_dir, paths + ".mp4")
            else:
                video_path = os.path.join(self.data_root_dir, paths)
            sample_info = {
                "path": video_path
            }
            index = vid_id
            retries = 5 if self.split == "train" else 10   # 1
            for retry in range(retries):
                try:
                    data, _, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, index, sample_info["path"]
                    ))
            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                data["video"] = self.transform(data["video"]) # C, T, H, W = 3, 16, 240, 320, RGB    
            return data["video"].permute(1,0,2,3), vid_id
    
    def get_seq_query(self, label, idx=-1):
        """Gets a single video sequence for a meta batch.  """
        c = self.split_few_shot
        if self.cfg.TRAIN.META_BATCH:
            paths, vid_id = c.get_rand_vid(label, idx) 
            if self.dataset_name == 'Video_few_shot':
                video_path = os.path.join(self.data_root_dir, paths + ".mp4")
            else:
                video_path = os.path.join(self.data_root_dir, paths)
            sample_info = {
                "path": video_path
            }
            index = vid_id
            retries = 5 if self.split == "train" else 10   # 1
            for retry in range(retries):
                try:
                    data, _, success = self.decode(
                        sample_info, index, num_clips_per_video=self.num_clips_per_video if hasattr(self, 'num_clips_per_video') else 1
                    )
                    break
                except Exception as e:
                    success = False
                    traceback.print_exc()
                    logger.warning("Error at decoding. {}/{}. Vid index: {}, Vid path: {}".format(
                        retry+1, retries, index, sample_info["path"]
                    ))
            if not success:
                logger.info("Error at decoding. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if data["video"].numel() == 0:
                logger.info("data[video].numel()=0. Vid index: {}, Vid path: {}".format(
                    index, sample_info["path"]))
                return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
            if self.gpu_transform:
                for k, v in data.items():
                    data[k] = v.cuda(non_blocking=True)
            if self._pre_transformation_config_required:
                self._pre_transformation_config()
            if "flow" in data.keys() and "video" in data.keys():
                data = self.transform(data)
            elif "video" in data.keys():  # [8, 240, 428, 3] --> [3, 8, 224, 224]
                data["video"] = self.transform(data["video"]) # C, T, H, W = 3, 16, 240, 320, R
            return data["video"].permute(1,0,2,3), vid_id

    def __len__(self):
        if hasattr(self.cfg.TRAIN, "META_BATCH") and self.split == 'train' and self.cfg.TRAIN.META_BATCH:
            return self.cfg.TRAIN.NUM_SAMPLES
        elif hasattr(self.cfg.TRAIN, "NUM_TEST_TASKS") and self.cfg.TRAIN.NUM_TEST_TASKS:
            return self.cfg.TRAIN.NUM_TEST_TASKS
        else:
            return len(self.split_few_shot)
    
    def _config_transform(self):
        self.transform = None
        if self.split == 'train':
            std_transform_list_query = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]  
            if hasattr(self.cfg.AUGMENTATION, "RANDOM_FLIP") and self.cfg.AUGMENTATION.RANDOM_FLIP:
                std_transform_list = [
                transforms.ToTensorVideo(),
                transforms.RandomHorizontalFlipVideo(),
                KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                    crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                ),]
            else:
                std_transform_list = [
                    transforms.ToTensorVideo(),
                    KineticsResizedCropFewshot(
                        short_side_range = [self.cfg.DATA.TRAIN_JITTER_SCALES[0], self.cfg.DATA.TRAIN_JITTER_SCALES[1]],
                        crop_size = self.cfg.DATA.TRAIN_CROP_SIZE,
                    ),
                ]
            # Add color aug
            if self.cfg.AUGMENTATION.COLOR_AUG:
                std_transform_list.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query.append(
                    ColorJitter(
                        brightness=self.cfg.AUGMENTATION.BRIGHTNESS,
                        contrast=self.cfg.AUGMENTATION.CONTRAST,
                        saturation=self.cfg.AUGMENTATION.SATURATION,
                        hue=self.cfg.AUGMENTATION.HUE,
                        grayscale=self.cfg.AUGMENTATION.GRAYSCALE,
                        consistent=self.cfg.AUGMENTATION.CONSISTENT,
                        shuffle=self.cfg.AUGMENTATION.SHUFFLE,
                        gray_first=self.cfg.AUGMENTATION.GRAY_FIRST,
                        is_split=self.cfg.AUGMENTATION.IS_SPLIT
                    ),
                )
            std_transform_list_query += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
                ]
            if hasattr(self.cfg.AUGMENTATION, "NO_RANDOM_ERASE") and self.cfg.AUGMENTATION.NO_RANDOM_ERASE:
                std_transform_list += [
                    transforms.NormalizeVideo(
                        mean=self.cfg.DATA.MEAN,
                        std=self.cfg.DATA.STD,
                        inplace=True
                    ),
                ]
            else:
                std_transform_list += [
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                ),
                RandomErasing(self.cfg)
                ]
            self.transform = Compose(std_transform_list)
            self.transform_query = Compose(std_transform_list_query)
        elif self.split == 'val' or self.split == 'test':
            idx = -1
            if hasattr(self.cfg.DATA, "TEST_CENTER_CROP"):
                idx = self.cfg.DATA.TEST_CENTER_CROP
            if isinstance(self.cfg.DATA.TEST_SCALE, list):
                self.resize_video = KineticsResizedCropFewshot(
                    short_side_range = [self.cfg.DATA.TEST_SCALE[0], self.cfg.DATA.TEST_SCALE[1]],
                    crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                    idx = idx
                )   # KineticsResizedCrop
            else:
                self.resize_video = KineticsResizedCropFewshot(
                        short_side_range = [self.cfg.DATA.TEST_SCALE, self.cfg.DATA.TEST_SCALE],
                        crop_size = self.cfg.DATA.TEST_CROP_SIZE,
                        idx = idx
                    )   # KineticsResizedCrop
            std_transform_list = [
                transforms.ToTensorVideo(),
                self.resize_video,
                transforms.NormalizeVideo(
                    mean=self.cfg.DATA.MEAN,
                    std=self.cfg.DATA.STD,
                    inplace=True
                )
            ]
            self.transform = Compose(std_transform_list)

    def _pre_transformation_config(self):
        """
            Set transformation parameters if required.
        """
        self.resize_video.set_spatial_index(self.spatial_idx)