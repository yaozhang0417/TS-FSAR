import os
import math
import datetime
from collections import defaultdict, deque
from time import perf_counter
from typing import Optional
import numpy as np
import torch
import psutil
import utils.logging as logging

logger = logging.get_logger(__name__)

# ------------------------ Timer ------------------------
class Timer:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._start = perf_counter()
        self._paused: Optional[float] = None
        self._total_paused = 0
        self._count_start = 1

    def pause(self) -> None:
        if self._paused is not None:
            raise ValueError("Trying to pause a Timer that is already paused!")
        self._paused = perf_counter()

    def resume(self) -> None:
        if self._paused is None:
            raise ValueError("Trying to resume a Timer that is not paused!")
        self._total_paused += perf_counter() - self._paused
        self._paused = None
        self._count_start += 1

    def seconds(self) -> float:
        if self._paused is not None:
            end_time: float = self._paused
        else:
            end_time = perf_counter()
        return end_time - self._start - self._total_paused

# ------------------------ Checkpoint ------------------------
def make_checkpoint_dir(path_to_job):
    checkpoint_dir = os.path.join(path_to_job, "checkpoints")
    if not os.path.exists(checkpoint_dir):
        try:
            os.makedirs(checkpoint_dir)
        except Exception:
            pass
    return checkpoint_dir

# ------------------------ Misc ------------------------
def check_nan_losses(loss):
    if math.isnan(loss):
        raise RuntimeError("ERROR: Got NaN losses {}".format(datetime.datetime.now()))

def params_count(model):
    return np.sum([p.numel() for p in model.parameters()]).item()

def gpu_mem_usage():
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3

def cpu_mem_usage():
    vram = psutil.virtual_memory()
    usage = (vram.total - vram.available) / 1024 ** 3
    total = vram.total / 1024 ** 3
    return usage, total

def get_num_gpus(cfg):
    return cfg.NUM_GPUS

def _get_model_analysis_input(cfg, use_train_input):
    rgb_dimension = 3
    if use_train_input:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_INPUT_FRAMES,
            cfg.DATA.TRAIN_CROP_SIZE,
            cfg.DATA.TRAIN_CROP_SIZE,
        )
    else:
        input_tensors = torch.rand(
            rgb_dimension,
            cfg.DATA.NUM_INPUT_FRAMES,
            cfg.DATA.TEST_CROP_SIZE,
            cfg.DATA.TEST_CROP_SIZE,
        )
    model_inputs = input_tensors.unsqueeze(0)
    if cfg.NUM_GPUS:
        model_inputs = model_inputs.cuda(non_blocking=True)
    inputs = {"video": model_inputs}
    return inputs

def get_model_stats(model, cfg, mode, use_train_input):
    assert mode in ["flop", "activation"], "'{}' not supported for model analysis".format(mode)
    try:
        from fvcore.nn.activation_count import activation_count
        from fvcore.nn.flop_count import flop_count
        if mode == "flop":
            model_stats_fun = flop_count
            from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS
            _DEFAULT_SUPPORTED_OPS["aten::batch_norm"] = None
        elif mode == "activation":
            model_stats_fun = activation_count
            from fvcore.nn.activation_count import _DEFAULT_SUPPORTED_OPS
        model_mode = model.training
        model.eval()
        inputs = _get_model_analysis_input(cfg, use_train_input)
        count_dict, _ = model_stats_fun(model, inputs, _DEFAULT_SUPPORTED_OPS)
        count = sum(count_dict.values())
        model.train(model_mode)
    except:
        count = None
    return count

def log_model_info(model, cfg, use_train_input=True):
    logger.info("Model:\n{}".format(model))
    logger.info("Params: {:,}".format(params_count(model)))
    logger.info("Mem: {:,} MB".format(gpu_mem_usage()))
    flops = get_model_stats(model, cfg, "flop", use_train_input)
    activations = get_model_stats(model, cfg, "activation", use_train_input)
    if flops is not None:
        logger.info("Flops: {:,} G".format(flops))
    if activations is not None:
        logger.info("Activations: {:,} M".format(activations))
    logger.info("nvidia-smi")
    os.system("nvidia-smi")

# ------------------------ Registry ------------------------
class Registry(object):
    def __init__(self, table_name=""):
        self._entry_map = {}
        self.table_name = table_name

    def _register(self, name, entry):
        assert type(name) is str
        assert (name not in self._entry_map.keys()), "{} {} already registered.".format(
            self.table_name, name
        )
        self._entry_map[name] = entry
    
    def register(self):
        def reg(obj):
            name = obj.__name__
            self._register(name, obj)
            return obj
        return reg
    
    def get(self, name):
        return self._entry_map.get(name, None)
    
    def get_all_registered(self):
        return self._entry_map.keys()

# ------------------------ Meters ------------------------
class ScalarMeter(object):
    def __init__(self, window_size=10):
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        return np.median(self.deque)

    def get_win_avg(self):
        return np.mean(self.deque)

    def get_global_avg(self):
        return self.total / self.count

class TrainMeter(object):
    def __init__(self, epoch_iters, cfg):
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.num_top1_mis = 0
        self.num_samples = 0
        self.opts = defaultdict(ScalarMeter)

    def reset(self):
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.num_top1_mis = 0
        self.num_samples = 0
        self.opts = defaultdict(ScalarMeter)

    def iter_tic(self):
        self.iter_timer.reset()

    def iter_toc(self):
        self.iter_timer.pause()

    def update_stats(self, top1_err, loss, lr, mb_size, **kwargs):
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size
        for k,v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.opts[k].add_value(v)
        self.mb_top1_err.add_value(top1_err)
        self.num_top1_mis += top1_err * mb_size

    def log_iter_stats(self, cur_iter):
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        stats = {
            "_type": "train_iter",
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(gpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_win_median()
        stats["top1_err"] = self.mb_top1_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self):
        stats = {
            "_type": "train_epoch",
            "lr": self.lr,
            "gpu_mem": "{:.2f} GB".format(gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f} GB".format(*cpu_mem_usage()),
        }
        for k,v in self.opts.items():
            stats[k] = v.get_global_avg()
        top1_err = self.num_top1_mis / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats["top1_err"] = top1_err
        stats["loss"] = avg_loss
        logging.log_json_stats(stats)
