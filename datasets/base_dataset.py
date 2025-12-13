import os
import random
import torch
import torchvision
import torch.utils.data
import torch.utils.dlpack as dlpack
import utils.logging as logging
import abc
import decord
import traceback
from decord import VideoReader
decord.bridge.set_bridge('native')
from torchvision.transforms import Compose

logger = logging.get_logger(__name__)

class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, split):
        """
        For initialization of the dataset, the global cfg and the split need to provided.
        Args:
            cfg     (Config): The global config object.
            split   (str): The split, e.g., "train", "val", "test"
        """
        self.cfg            = cfg
        self.split          = split
        self.data_root_dir  = cfg.DATA.DATA_ROOT_DIR
        self.anno_dir       = cfg.DATA.ANNO_DIR
        self._num_clips = 1
        if self.split in ["train", "val"]:
            self.dataset_name = cfg.TRAIN.DATASET
        elif self.split in ["test"]:
            self.dataset_name = cfg.TEST.DATASET
        else:
            raise NotImplementedError("Split not supported")
        self._num_frames = cfg.DATA.NUM_INPUT_FRAMES
        self._sampling_rate = cfg.DATA.SAMPLING_RATE
        self.gpu_transform = cfg.AUGMENTATION.USE_GPU       # whether or not to perform the transform on GPU
        self.decode = self._decode_video                    # decode function, decode videos by default
        # if set to true, _pre_transformation_config will be called before every transformations
        # this is used in the testset, where cropping positions are set for the controlled crop
        self._pre_transformation_config_required = False    
        self._construct_dataset(cfg)
        self._config_transform()

    @abc.abstractmethod
    def _get_dataset_list_name(self):
        raise NotImplementedError

    @abc.abstractmethod
    def _get_sample_info(self, index):
        raise NotImplementedError

    @abc.abstractmethod
    def _config_transform(self):
        self.transform = Compose([])
        raise NotImplementedError

    @abc.abstractmethod
    def _pre_transformation_config(self):
        raise NotImplementedError

    def _get_video_frames_list(self, vid_length, vid_fps, clip_idx, random_sample=True):
        """
        Returns the list of frame indexes in the video for decoding. 
        Args:
            vid_length (int): video length
            clip_idx (int): clip index, -1 if random sampling (interval based sampling)
            num_clips (int): overall number of clips for clip_idx != -1 (interval based sampling) 
            num_frames (int): number of frames to sample 
            interval (int): the step size for interval based sampling (interval based sampling)
            random_sample (int): whether to randomly sample one frame from each segment (segment based sampling)
        Returns:
            frame_id_list (list): indicates which frames to sample from the video
        """
        if self.cfg.DATA.SAMPLING_MODE == "interval_based":
            return self._interval_based_sampling(vid_length, vid_fps, clip_idx, self._num_frames, self._sampling_rate)
        elif self.cfg.DATA.SAMPLING_MODE == "segment_based":
            return self._segment_based_sampling(vid_length, self._num_frames, random_sample)
        else:
            raise NotImplementedError

    def _construct_dataset(self, cfg):
        self._samples = []
        self._spatial_temporal_index = []
        dataset_list_name = self._get_dataset_list_name()
        list_path = os.path.join(self.anno_dir, dataset_list_name)
        if not os.path.isfile(list_path):
            raise ValueError(f"Dataset list not found: {list_path}")
        logger.info(f"Loading list {list_path} split={self.split}")
        # Load file list
        if list_path.endswith(".csv"):
            import pandas
            lines = pandas.read_csv(list_path).values.tolist()
        elif list_path.endswith(".json"):
            import json
            with open(list_path, "r") as f:
                lines = json.load(f)
        else:
            with open(list_path, "r") as f:
                lines = [line.strip() for line in f.readlines()]
        for line in lines:
            for idx in range(self._num_clips):
                self._samples.append(line)
                self._spatial_temporal_index.append(idx)
        logger.info(f"Dataset loaded: {len(self._samples)} samples.")

    def _read_video(self, video_path, index):
        """
        Wrapper for downloading the video and generating the VideoReader object for reading the video. 
        Args: 
            video_path (str): video path to read the video from. Can in OSS form or in local hard drives.
            index      (int):  for debug.
        Returns:
            vr              (VideoReader):  VideoReader object wrapping the video.
            success         (bool):         flag for the indication of success or not.
        """
        try:
            vr = VideoReader(video_path)
            return vr, [], True
        except Exception as e:
            print(f"[Error] Cannot read video {video_path}")
            traceback.print_exc()
            return None, [], False

    def _decode_video(self, sample_info, index, num_clips_per_video=1):
        """
        Decodes the video given the sample info.
        Args: 
            sample_info         (dict): containing the "path" key specifying the location of the video.
            index               (int):  for debug.
            num_clips_per_video (int):  number of clips to be decoded from each video. set to 2 for contrastive learning and 1 for others.
        Returns:
            data            (dict): key "video" for the video data.
            file_to_remove  (list): list of temporary files to be deleted or BytesIO objects to be closed.
            success         (bool): flag for the indication of success or not.
        """
        path = sample_info["path"]
        vr, _, success =  self._read_video(path, index)
        if not success:
            return None, [], False
        if self.split == "train":
            clip_idx = -1
            self.spatial_idx = -1
        elif self.split == "val":
            clip_idx = -1
            self.spatial_idx = 0
        elif self.split == "test":
            clip_idx = self._spatial_temporal_index[index]
            self.spatial_idx = 0
        frame_list= []
        for _ in range(num_clips_per_video):
            # for each clip in the video, 
            # a list is generated before decoding the specified frames from the video
            list_ = self._get_video_frames_list(
                len(vr),
                vr.get_avg_fps(),
                clip_idx,
                random_sample=True if self.split=="train" else False 
            )
            frames = None
            frames = dlpack.from_dlpack(vr.get_batch(list_).to_dlpack()).clone()
            frame_list.append(frames)
        frames = torch.stack(frame_list)
        if num_clips_per_video == 1:
            frames = frames.squeeze(0)
        return {"video": frames}, [], True

    def __getitem__(self, index):
        """
        Gets the specified data.
        Args:
            index (int): the index of the data in the self._samples list.
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
        sample_info = self._get_sample_info(index)
        # decode the data
        retries = 1 if self.split == "train" else 10
        for retry in range(retries):
            try:
                data, file_to_remove, success = self.decode(
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
            return self.__getitem__(index - 1) if index != 0 else self.__getitem__(index + 1)
        if self.gpu_transform:
            for k, v in data.items():
                data[k] = v.cuda(non_blocking=True)
        if self._pre_transformation_config_required:
            self._pre_transformation_config()
        return data, {}, index, {}
    
    def __len__(self):
        """
        Returns the number of samples.
        """
        if hasattr(self.cfg.TRAIN, "NUM_SAMPLES") and self.split == 'train':
            return self.cfg.TRAIN.NUM_SAMPLES
        else:
            return len(self._samples)
    
    def _interval_based_sampling(self, vid_length, vid_fps, clip_idx, num_frames, interval):
        if num_frames == 1:
            index = [random.randint(0, vid_length-1)]
        else:
            if self.split == "train" and hasattr(self.cfg.DATA, "SAMPLING_RATE_TRAIN"):
                interval = self.cfg.DATA.SAMPLING_RATE_TRAIN
                clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS
            elif hasattr(self.cfg.DATA, "SAMPLING_RATE_TEST") and self.cfg.DATA.SAMPLING_RATE_TEST>40:
                interval = vid_length//num_frames
                clip_length = vid_length//num_frames * num_frames
                index = [random.randint(ind*interval, ind*interval+interval-1) for ind in range(num_frames)]
                return index
            elif self.cfg.DATA.SAMPLING_RATE >40:  # SAMPLING_RATE_TEST
                interval = vid_length//num_frames
                clip_length = vid_length//num_frames * num_frames
                index = [random.randint(ind*interval, ind*interval+interval-1) for ind in range(num_frames)]
                return index
            else:
            # transform FPS
                clip_length = num_frames * interval * vid_fps / self.cfg.DATA.TARGET_FPS
            if clip_length > vid_length:
                clip_length = vid_length//num_frames * num_frames
            max_idx = max(vid_length - clip_length + 1, 0)
            if clip_idx == -1: # random sampling
                start_idx = random.uniform(0, max_idx)
            else:
                start_idx = max_idx / 2
            end_idx = start_idx + clip_length - interval
            index = torch.linspace(start_idx, end_idx, num_frames)
            index = torch.clamp(index, 0, vid_length-1).long()
        return index

    def _segment_based_sampling(self, vid_length, num_frames, random_sample):
        """
        Generates the frame index list using segment based sampling.
        Args:
            vid_length    (int):  the length of the whole video (valid selection range).
            num_clips     (int):  the total clips to be sampled from each video. 
            num_frames    (int):  number of frames in each sampled clips.
            random_sample (bool): whether or not to randomly sample from each segment. True for train and False for test.
        Returns:
            index (tensor): the sampled frame indexes
        """
        index = torch.zeros(num_frames)
        index_range = torch.linspace(0, vid_length, num_frames+1)
        for idx in range(num_frames):
            if random_sample:
                index[idx] = random.uniform(index_range[idx], index_range[idx+1])
            else:
                index[idx] = (index_range[idx] + index_range[idx+1]) / 2
        index = torch.round(torch.clamp(index, 0, vid_length-1)).long()
        return index