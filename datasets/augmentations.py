import torch
import math
from torchvision.transforms import Lambda, Compose
import random
import numbers

class ColorJitter(object):
    """
    Randomly change the brightness, contrast and saturation of an image.
    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.
        grayscale (float): possibility to transform the video to grayscale. 
            Should have a value range of [0, 1]
        consistent  (bool): indicates whether or not to keep all the color transformations consistent for all the frames.
        shuffle     (bool): indicates whether or not to shuffle the sequence of the augmentations.
        gray_first  (bool): indicates whether or not to put grayscale transform first.
    """
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, grayscale=0, consistent=False, shuffle=True, gray_first=True, is_split=False):
        self.brightness = self._check_input(brightness, 'brightness')
        self.contrast = self._check_input(contrast, 'contrast')
        self.saturation = self._check_input(saturation, 'saturation')
        self.hue = self._check_input(hue, 'hue', center=0, bound=(-0.5, 0.5),
                                     clip_first_on_zero=False)
        self.grayscale = grayscale
        self.consistent = consistent
        self.shuffle = shuffle
        self.gray_first = gray_first
        self.is_split = is_split

    def _check_input(self, value, name, center=1, bound=(0, float('inf')), clip_first_on_zero=True):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError("If {} is a single number, it must be non negative.".format(name))
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError("{} should be a single number or a list/tuple with lenght 2.".format(name))
        if value[0] == value[1] == center:
            value = None
        return value

    def _get_transform(self, T, device):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Arg:
            T (int): number of frames. Used when consistent = False.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []
        if self.brightness is not None:
            if self.consistent:
                brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            else:
                brightness_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.brightness[0], self.brightness[1])
            transforms.append(Lambda(lambda frame: adjust_brightness(frame, brightness_factor)))
        if self.contrast is not None:
            if self.consistent:
                contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            else:
                contrast_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.contrast[0], self.contrast[1])
            transforms.append(Lambda(lambda frame: adjust_contrast(frame, contrast_factor)))
        if self.saturation is not None:
            if self.consistent:
                saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            else:
                saturation_factor = torch.empty([1, T, 1, 1], device=device).uniform_(self.saturation[0], self.saturation[1])
            transforms.append(Lambda(lambda frame: adjust_saturation(frame, saturation_factor)))
        if self.hue is not None:
            if self.consistent:
                hue_factor = random.uniform(self.hue[0], self.hue[1])
            else:
                hue_factor = torch.empty([T, 1, 1], device=device).uniform_(self.hue[0], self.hue[1])
            transforms.append(Lambda(lambda frame: adjust_hue(frame, hue_factor)))
        if self.shuffle:
            random.shuffle(transforms)
        if random.uniform(0, 1) < self.grayscale:
            gray_transform = Lambda(lambda frame: rgb_to_grayscale(frame, split=self.is_split))
            if self.gray_first: 
                transforms.insert(0, gray_transform)
            else:
                transforms.append(gray_transform)
        transform = Compose(transforms)
        return transform

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (C, T, H, W)
        Return:
            clip (torch.tensor): Size is (C, T, H, W)
        """
        raw_shape = clip.shape #(C, T, H, W)
        device = clip.device
        T = raw_shape[1]
        transform = self._get_transform(T, device)
        clip = transform(clip)
        assert clip.shape == raw_shape
        return clip #(C, T, H, W)

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', grayscale={0})'.format(self.grayscale)
        return format_string

def _is_tensor_a_torch_image(input):
    return input.ndim >= 2

def _blend(img1, img2, ratio):
    bound = 1 if img1.dtype in [torch.half, torch.float32, torch.float64] else 255
    return (ratio * img1 + (1 - ratio) * img2).clamp(0, bound).to(img1.dtype)

def rgb_to_grayscale(img, split=False):
    """Convert the given RGB Image Tensor to Grayscale.
    For RGB to Grayscale conversion, ITU-R 601-2 luma transform is performed which
    is L = R * 0.2989 + G * 0.5870 + B * 0.1140
    Args:
        img (Tensor): Image to be converted to Grayscale in the form [C, H, W].
    Returns:
        Tensor: Grayscale image.
        Args:
            clip (torch.tensor): Size is (T, H, W, C)
        Return:
            clip (torch.tensor): Size is (T, H, W, C)
    """
    orig_dtype = img.dtype
    rgb_convert = torch.tensor([0.299, 0.587, 0.114])
    if split:
        rgb_convert *= 0
        channel = random.randint(0, 2)
        rgb_convert[channel] = 1
    assert img.shape[0] == 3, "First dimension need to be 3 Channels"
    if img.is_cuda:
        rgb_convert = rgb_convert.to(img.device)
    img = img.float().permute(1,2,3,0).matmul(rgb_convert).to(orig_dtype)
    return torch.stack([img, img, img], 0)

def _rgb2hsv(img):
    r, g, b = img.unbind(0)
    maxc, _ = torch.max(img, dim=0)
    minc, _ = torch.min(img, dim=0)
    eqc = maxc == minc
    cr = maxc - minc
    s = cr / torch.where(eqc, maxc.new_ones(()), maxc)
    cr_divisor = torch.where(eqc, maxc.new_ones(()), cr)
    rc = (maxc - r) / cr_divisor
    gc = (maxc - g) / cr_divisor
    bc = (maxc - b) / cr_divisor
    hr = (maxc == r) * (bc - gc)
    hg = ((maxc == g) & (maxc != r)) * (2.0 + rc - bc)
    hb = ((maxc != g) & (maxc != r)) * (4.0 + gc - rc)
    h = (hr + hg + hb)
    h = torch.fmod((h / 6.0 + 1.0), 1.0)
    return torch.stack((h, s, maxc))

def _hsv2rgb(img):
    l = len(img.shape)
    h, s, v = img.unbind(0)
    i = torch.floor(h * 6.0)
    f = (h * 6.0) - i
    i = i.to(dtype=torch.int32)
    p = torch.clamp((v * (1.0 - s)), 0.0, 1.0)
    q = torch.clamp((v * (1.0 - s * f)), 0.0, 1.0)
    t = torch.clamp((v * (1.0 - s * (1.0 - f))), 0.0, 1.0)
    i = i % 6
    if l == 3:
        tmp = torch.arange(6)[:, None, None]
    elif l == 4:
        tmp = torch.arange(6)[:, None, None, None]
    if img.is_cuda:
        tmp = tmp.to(img.device)
    mask = i == tmp #(H, W) == (6, H, W)
    a1 = torch.stack((v, q, p, p, t, v))
    a2 = torch.stack((t, v, v, q, p, p))
    a3 = torch.stack((p, p, t, v, v, q))
    a4 = torch.stack((a1, a2, a3)) #(3, 6, H, W)
    if l == 3:
        return torch.einsum("ijk, xijk -> xjk", mask.to(dtype=img.dtype), a4) #(C, H, W)
    elif l == 4:
        return torch.einsum("itjk, xitjk -> xtjk", mask.to(dtype=img.dtype), a4) #(C, T, H, W)

def adjust_brightness(img, brightness_factor):
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')
    return _blend(img, torch.zeros_like(img), brightness_factor)

def adjust_contrast(img, contrast_factor):
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')
    mean = torch.mean(rgb_to_grayscale(img).to(torch.float), dim=(-4, -2, -1), keepdim=True)
    return _blend(img, mean, contrast_factor)

def adjust_saturation(img, saturation_factor):
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')
    return _blend(img, rgb_to_grayscale(img), saturation_factor)

def adjust_hue(img, hue_factor):
    """Adjust hue of an image.
    The image hue is adjusted by converting the image to HSV and
    cyclically shifting the intensities in the hue channel (H).
    The image is then converted back to original image mode.
    `hue_factor` is the amount of shift in H channel and must be in the
    interval `[-0.5, 0.5]`.
    Args:
        img (Tensor): Image to be adjusted. Image type is either uint8 or float.
        hue_factor (float):  How much to shift the hue channel. Should be in
            [-0.5, 0.5]. 0.5 and -0.5 give complete reversal of hue channel in
            HSV space in positive and negative direction respectively.
            0 means no shift. Therefore, both -0.5 and 0.5 will give an image
            with complementary colors while 0 gives the original image.
    Returns:
         Tensor: Hue adjusted image.
    """
    if isinstance(hue_factor, float) and  not(-0.5 <= hue_factor <= 0.5):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    elif isinstance(hue_factor, torch.Tensor) and not ((-0.5 <= hue_factor).sum() == hue_factor.shape[0] and (hue_factor <= 0.5).sum() == hue_factor.shape[0]):
        raise ValueError('hue_factor ({}) is not in [-0.5, 0.5].'.format(hue_factor))
    if not _is_tensor_a_torch_image(img):
        raise TypeError('tensor is not a torch image.')
    orig_dtype = img.dtype
    if img.dtype == torch.uint8:
        img = img.to(dtype=torch.float32) / 255.0
    img = _rgb2hsv(img)
    h, s, v = img.unbind(0)
    h += hue_factor
    h = h % 1.0
    img = torch.stack((h, s, v))
    img_hue_adj = _hsv2rgb(img)
    if orig_dtype == torch.uint8:
        img_hue_adj = (img_hue_adj * 255.0).to(dtype=orig_dtype)
    return img_hue_adj

class KineticsResizedCropFewshot(object):
    def __init__(
        self,
        short_side_range,
        crop_size,
        idx = -1
    ):  
        self.idx = idx
        self.short_side_range = short_side_range
        self.crop_size = int(crop_size)
    
    def _get_controlled_crop(self, clip): 
        if len(self.short_side_range)>1:
            new_clip_width = self.short_side_range[1]
            new_clip_height = self.short_side_range[0]
            new_clip = torch.nn.functional.interpolate(
                clip, size=(self.short_side_range[0], self.short_side_range[1]), mode="bilinear"  
            )
        else:
            new_clip_width = self.short_side_range[0]
            new_clip_height = self.short_side_range[0]
            new_clip = torch.nn.functional.interpolate(
                clip, size=(self.short_side_range[0], self.short_side_range[0]), mode="bilinear"
            )
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        x = x_max // 2
        y = y_max // 2
        return new_clip[:, :, y:y+self.crop_size, x:x+self.crop_size]

    def _get_random_crop(self, clip):
        if len(self.short_side_range)>1:
            new_clip_width = self.short_side_range[1]
            new_clip_height = self.short_side_range[0]
            new_clip = torch.nn.functional.interpolate(
                clip, size=(self.short_side_range[0], self.short_side_range[1]), mode="bilinear"
            )
        else:
            new_clip_width = self.short_side_range[0]
            new_clip_height = self.short_side_range[0]
            new_clip = torch.nn.functional.interpolate(
                clip, size=(self.short_side_range[0], self.short_side_range[0]), mode="bilinear"
            )
        x_max = int(new_clip_width - self.crop_size)
        y_max = int(new_clip_height - self.crop_size)
        x = int(random.uniform(0, x_max))
        y = int(random.uniform(0, y_max))
        return new_clip[:, :, y:y+self.crop_size, x:x+self.crop_size]

    def set_spatial_index(self, idx):
        self.idx = idx

    def __call__(self, clip):
        if self.idx == -1:
            return self._get_random_crop(clip)
        else:
            return self._get_controlled_crop(clip)

def _get_pixels(per_pixel, rand_color, patch_size, dtype=torch.float32, device='cuda'):
    # NOTE I've seen CUDA illegal memory access errors being caused by the normal_()
    # paths, flip the order so normal is run on CPU if this becomes a problem
    # Issue has been fixed in master https://github.com/pytorch/pytorch/issues/19508
    if per_pixel:
        return torch.empty(patch_size, dtype=dtype, device=device).normal_()
    elif rand_color:
        return torch.empty((patch_size[0], 1, 1, 1), dtype=dtype, device=device).normal_()
    else:
        return torch.zeros((patch_size[0], 1, 1, 1), dtype=dtype, device=device)


class RandomErasing:
    """ Randomly selects a rectangle region in an image and erases its pixels.
        This variant of RandomErasing is intended to be applied to either a batch
        or single image tensor after it has been normalized by dataset mean and std.
    Args:
         probability: Probability that the Random Erasing operation will be performed.
         min_area: Minimum percentage of erased area wrt input image area.
         max_area: Maximum percentage of erased area wrt input image area.
         min_aspect: Minimum aspect ratio of erased area.
         mode: pixel color mode, one of 'const', 'rand', or 'pixel'
            'const' - erase block is constant color of 0 for all channels
            'rand'  - erase block is same per-channel random (normal) color
            'pixel' - erase block is per-pixel random (normal) color
        max_count: maximum number of erasing blocks per image, area per box is scaled by count.
            per-image count is randomly chosen between 1 and this value.
    """
    def __init__(self, cfg,):
        """
        Args:
            cfg (Config): global config object. 
        """
        self.enable                     = cfg.AUGMENTATION.RANDOM_ERASING.ENABLE
        self.probability                = cfg.AUGMENTATION.RANDOM_ERASING.PROB
        self.min_area, self.max_area    = cfg.AUGMENTATION.RANDOM_ERASING.AREA_RANGE
        min_aspect                      = cfg.AUGMENTATION.RANDOM_ERASING.MIN_ASPECT
        max_aspect                      = 1 / min_aspect
        self.log_aspect_ratio           = (math.log(min_aspect), math.log(max_aspect))
        self.min_count, self.max_count  = cfg.AUGMENTATION.RANDOM_ERASING.COUNT
        self.num_splits                 = cfg.AUGMENTATION.RANDOM_ERASING.NUM_SPLITS
        mode                            = cfg.AUGMENTATION.RANDOM_ERASING.MODE.lower()
        self.rand_color = False
        self.per_pixel = False
        if mode == 'rand':
            self.rand_color = True  # per block random normal
        elif mode == 'pixel':
            self.per_pixel = True  # per pixel random normal
        else:
            assert not mode or mode == 'const'

    def _erase(self, img, chan, num_frames, img_h, img_w, dtype):
        if random.random() > self.probability:
            return
        area = img_h * img_w
        count = self.min_count if self.min_count == self.max_count else \
            random.randint(self.min_count, self.max_count)
        for _ in range(count):
            for attempt in range(10):
                target_area = random.uniform(self.min_area, self.max_area) * area / count
                aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))
                if w < img_w and h < img_h:
                    top = random.randint(0, img_h - h)
                    left = random.randint(0, img_w - w)
                    img[:, :, top:top + h, left:left + w] = _get_pixels(
                        self.per_pixel, self.rand_color, (chan, num_frames, h, w),
                        dtype=dtype, device=img.device)
                    break

    def __call__(self, input):
        if self.enable:
            if len(input.size()) == 4:
                self._erase(input, *input.size(), input.dtype)
            else:
                batch_size, chan, num_frames, img_h, img_w = input.size()
                # skip first slice of batch if num_splits is set (for clean portion of samples)
                batch_start = batch_size // self.num_splits if self.num_splits > 1 else 0
                for i in range(batch_start, batch_size):
                    self._erase(input[i], chan, num_frames, img_h, img_w, input.dtype)
        return input