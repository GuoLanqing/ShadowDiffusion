import argparse
import random
from pathlib import Path

import numpy as np
import torch


# to implement color jitter in
# https://github.com/google/portrait-shadow-manipulation/blob/master/utils.py
# converting tf version to pt version
def apply_tone_curve(image, gain=(0.5, 0.5, 0.5), is_rgb=True):
    """Apply tone perturbation to images.
    Tone curve jitter comes from Schlick's bias and gain.
    Schlick, Christophe. "Fast alternatives to Perlin’s bias and gain functions." Graphics Gems IV 4 (1994).
    Args:
      image: a 3D image tensor [C, H, W].
      gain: a tuple of length 3 that specifies the strength of the jitter per color channel.
      is_rgb: a bool that indicates whether input is grayscale (C=1) or rgb (C=3).
    Returns:
      3D tensor applied with a tone curve jitter, has the same size as input.
    """

    def getbias(x, bias):
        return x / ((1.0 / bias - 2.0) * (1.0 - x) + 1.0 + 1e-6)

    image_max = image.max()
    image /= image_max

    for i in range(3):
        im = image[i]
        mask = (im > 0.5).float()
        # from IPython import embed; embed(); exit();
        image[i] = getbias(im * 2.0, gain[i]) / 2.0 * (1.0 - mask) + \
            (getbias(im * 2.0 - 1.0, 1.0 - gain[i]) / 2.0 + 0.5) * mask
    return image * image_max


# https://mikio.hatenablog.com/entry/2018/09/10/213756
def rgb_to_srgb(value: torch.FloatTensor):
    cond = value <= 0.0031308
    value[cond] = (value * 12.92)[cond]
    value[~cond] = ((value ** (1.0 / 2.4)) * 1.055 - 0.055)[~cond]
    return value


def srgb_to_rgb(value: torch.FloatTensor):
    cond = value <= 0.04045
    value[cond] = (value / 12.92)[cond]
    value[~cond] = (((value + 0.055) / 1.055) ** 2.4)[~cond]
    return value


def fit_brightening_params(
    input: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    mask_thresh: float = 0.95,
    N_min: int = 16  # minimum num of valid pixels for fitting
):

    # assume 3-dimensional tensor with (C, H, W) for input and target
    # assume 2-dimensional tensor with (H, W) for mask and ids
    assert input.size()[1:] == target.size()[1:] == mask.size()
    assert input.size(0) == 3 and target.size(0) == 3
    assert 0.0 <= mask.min() and mask.max() <= 1.0

    # Use only (apparently shadowed) region to avoid errors around boundary
    default_w, default_b = [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]

    shadow = input.numpy()
    shadow_free = target.numpy()
    cond = (mask > mask_thresh)

    W = torch.ones_like(target).float()
    B = torch.zeros_like(target).float()

    ids = torch.zeros_like(mask)
    if len(torch.unique(ids[cond])) == 0:
        # shadow region is cropped out
        return torch.Tensor(default_w).float(), torch.Tensor(default_b).float()

    if cond.sum() < N_min:  # We don't fit line
        w, b = default_w, default_b
    else:
        coef_list = []
        for i in range(3):  # for each RGB channel
            if len(np.unique(shadow[i][cond])) < 2:
                # Difficult to fit;
                coef = [1.0, 0.0]
            # try:
            #     # coef = np.polyfit(
            #     #     shadow[i][cond], shadow_free[i][cond], 1)
            #     coef = np.polyfit(
            #         shadow[i][cond].flatten(),
            #         shadow_free[i][cond].flatten(),
            #         1
            #     )
            # except ValueError:
            #     # print(shadow[i][cond], shadow_free[i][cond])
            #     coef = [1.0, 0.0]
            coef = [1.0, 0.0]
            coef_list.append(coef)
        w, b = zip(*coef_list)
    w, b = torch.Tensor(w).float(), torch.Tensor(b).float()

    return w, b  # (3,), (3,)


def sample_darkening_params(opt: argparse.ArgumentParser):
    """
        Sample two points, and generate its slope
    """
    assert 0.0 <= opt.xmax_at_y_0 <= 1.0
    assert 0.0 <= opt.ymin_at_x_255 <= 1.0
    assert 0.0 <= opt.ymax_at_x_255 <= 1.0
    assert 1.0 <= opt.slope_max  # opt.slope_max < 1.0 is very `normal`

    while True:
        x1 = random.uniform(opt.xmin_at_y_0, opt.xmax_at_y_0)
        y1 = 0.0
        x2 = 1.0
        y2 = random.uniform(opt.ymin_at_x_255, opt.ymax_at_x_255)
        a = (y2 - y1) / (x2 - x1)  # Assume slope = const. for all channels
        if a < 1.0:
            break
    return a, x1, y1, x2, y2


def darken(x: torch.Tensor, opt: argparse.ArgumentParser):
    if opt.intercepts_mode == 'affine_unsync':
        # use affine model, but slopes are not shared for each channel
        a, x1_R, y1_R, _, _ = sample_darkening_params(opt)
        _, x1_G, y1_G, _, _ = sample_darkening_params(opt)
        _, x1_B, y1_B, _, _ = sample_darkening_params(opt)

        b = [y1_R - a * x1_R, y1_G - a * x1_G, y1_B - a * x1_B]
        b = torch.Tensor(b).view(3, 1, 1)
        b = b.repeat(1, x.size(1), x.size(2))
    elif opt.intercepts_mode == 'affine':
        a, x1, y1, x2, y2 = sample_darkening_params(opt)
        if opt.x_turb_sigma == 0.0:
            b = y1 - a * x1  # no shift in x_intercept in RGB channel
        else:
            # R and G are usually larger than G and R, respectively.
            # y_g = a * x + b
            x1_G = x1
            mu, sigma = opt.x_turb_mu, opt.x_turb_sigma
            x1_R = x1_G + np.random.normal(mu, sigma)
            x1_B = x1_G - np.random.normal(mu, sigma)

            b = [y1 - a * x1_R, y1 - a * x1_G, y1 - a * x1_B]
            b = torch.Tensor(b).view(3, 1, 1)
            b = b.repeat(1, x.size(1), x.size(2))
    elif opt.intercepts_mode == 'random_jitter':
        TONE_SIGMA = 0.15
        delta = torch.FloatTensor(size=(3,))
        # curve_gain = 0.5 + delta.uniform_(-TONE_SIGMA, TONE_SIGMA)
        curve_gain = 0.5 + delta.uniform_(0.0, TONE_SIGMA)  # just decrease
        x = srgb_to_rgb(x.clone())
        occl_x = apply_tone_curve(x, curve_gain)

        # solve AX = B
        out = torch.lstsq(
            occl_x.permute((1, 2, 0)).view(-1, 3),  # B
            x.permute((1, 2, 0)).view(-1, 3),  # A
        )
        ctm = out[0][:3]
        out = torch.mm(x.permute((1, 2, 0)).view(-1, 3), ctm)
        out = out.permute((1, 0)).view(x.size())
        out = rgb_to_srgb(out)
        return out
    elif opt.intercepts_mode == 'gamma_correction':
        return x ** (1.5 + 1.5 * random.random())
    else:
        raise NotImplementedError

    y = a * x + b
    y = torch.clamp(y, min=0.0, max=1.0)
    return y