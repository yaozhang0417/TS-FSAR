import torch
from torch.utils.data.sampler import RandomSampler
from datasets.collate_functions import COLLATE_FN_REGISTRY
from utils.tools import Registry

DATASET_REGISTRY = Registry("DATASET")

def build_loader(cfg, split):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (Configs): global config object. details in utils/config.py
        split (str): the split of the data loader. Options include `train`,
            `val`, `test`, and `submission`.
    Returns:
        loader object. 
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET#Video_few_shot
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val","test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)
    # Create a loader
    if hasattr(cfg.DATA_LOADER, "COLLATE_FN") and cfg.DATA_LOADER.COLLATE_FN is not None:
        collate_fn = COLLATE_FN_REGISTRY.get(cfg.DATA_LOADER.COLLATE_FN)(cfg)
    else:
        collate_fn = None
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=cfg.DATA_LOADER.NUM_WORKERS,
        pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
        drop_last=drop_last,
        collate_fn=collate_fn
    )
    return loader

def shuffle_dataset(loader, cur_epoch):
    """"
    Shuffles the sampler for the dataset.
    Args:
        loader      (loader):   data loader to perform shuffle.
        cur_epoch   (int):      number of the current epoch.
    """
    sampler = loader.sampler
    # RandomSampler handles shuffling automatically
    assert isinstance(
        sampler, RandomSampler
    ), "Sampler type '{}' not supported".format(type(sampler))
    
def build_dataset(dataset_name, cfg, split):
    """
    Builds a dataset according to the "dataset_name".
    Args:
        dataset_name (str):     the name of the dataset to be constructed.
        cfg          (Config):  global config object. 
        split        (str):     the split of the data loader.
    Returns:
        Dataset      (Dataset):    a dataset object constructed for the specified dataset_name.
    """
    name = dataset_name.capitalize()#Video_few_shot
    return DATASET_REGISTRY.get(name)(cfg, split)
