"""Image processing for PL images of wafers.

.. moduleauthor:: Neil Yager <neil.yager@btimaging.com>

"""

import os
import sys
import numpy as np
from scipy import ndimage
import pixel_ops
import cv2
from image_processing import ImageViewer
import scipy.stats
import image_processing as ip
import math
import parameters
import gc
import matplotlib.pylab as plt
import statsmodels.api as sm

struct = ndimage.generate_binary_structure(2, 1)

SAVE_FEATURES = False


# Wafer types
class WaferType:
    FULLY_IMPURE = 0
    TRANSITION = 1
    MIDDLE = 2
    EDGE = 3
    CORNER = 4
    BROKEN = 5
    UNKNOWN = 6
    types = ['fully_impure', 'transition', 'middle', 'edge', 'corner', 'broken', 'UNKNOWN']


# Coordinates of expanding circle (used for interpolation). Pre-computed
#  so they don't need to be computed every time on the fly
ys = [0, -1, 0, 0, 1, -1, -1, 1, 1, -2, 0, 0, 2, -2, -2, -1, -1, 1, 1, 2, 2,
      -2, -2, 2, 2, -3, 0, 0, 3, -3, -3, -1, -1, 1, 1, 3, 3, -3, -3, -2, -2,
      2, 2, 3, 3, -4, 0, 0, 4, -4, -4, -1, -1, 1, 1, 4, 4, -3, -3, 3, 3,
      -4, -4, -2, -2, 2, 2, 4, 4, -5, -4, -4, -3, -3, 0, 0, 3, 3, 4, 4, 5,
      -5, -5, -1, -1, 1, 1, 5, 5, -5, -5, -2, -2, 2, 2, 5, 5, -4, -4, 4, 4,
      -5, -5, -3, -3, 3, 3, 5, 5, -6, 0, 0, 6, -6, -6, -1, -1, 1, 1, 6, 6,
      -6, -6, -2, -2, 2, 2, 6, 6, -5, -5, -4, -4, 4, 4, 5, 5, -6, -6, -3, -3,
      3, 3, 6, 6, -7, 0, 0, 7, -7, -7, -5, -5, -1, -1, 1, 1, 5, 5, 7, 7,
      -6, -6, -4, -4, 4, 4, 6, 6, -7, -7, -2, -2, 2, 2, 7, 7, -7, -7, -3, -3,
      3, 3, 7, 7, -6, -6, -5, -5, 5, 5, 6, 6, -8, 0, 0, 8, -8, -8, -7, -7,
      -4, -4, -1, -1, 1, 1, 4, 4, 7, 7, 8, 8, -8, -8, -2, -2, 2, 2, 8, 8,
      -6, -6, 6, 6, -8, -8, -3, -3, 3, 3, 8, 8, -7, -7, -5, -5, 5, 5, 7, 7,
      -8, -8, -4, -4, 4, 4, 8, 8, -9, 0, 0, 9, -9, -9, -1, -1, 1, 1, 9, 9,
      -7, -7, -6, -6, 6, 6, 7, 7, -9, -9, -2, -2, 2, 2, 9, 9, -8, -8, -5, -5,
      5, 5, 8, 8, -9, -9, -3, -3, 3, 3, 9, 9, -9, -9, -4, -4, 4, 4, 9, 9,
      -7, -7, 7, 7, -10, -8, -8, -6, -6, 0, 0, 6, 6, 8, 8, 10, -10, -10, -1, -1,
      1, 1, 10, 10, -10, -10, -2, -2, 2, 2, 10, 10, -9, -9, -5, -5, 5, 5, 9, 9,
      -10, -10, -3, -3, 3, 3, 10, 10, -8, -8, -7, -7, 7, 7, 8, 8, -10, -10, -4, -4,
      4, 4, 10, 10, -9, -9, -6, -6, 6, 6, 9, 9, -11, 0, 0, 11, -11, -11, -1, -1,
      1, 1, 11, 11, -11, -11, -2, -2, 2, 2, 11, 11, -10, -10, -5, -5, 5, 5, 10, 10,
      -8, -8, 8, 8, -11, -11, -9, -9, -7, -7, -3, -3, 3, 3, 7, 7, 9, 9, 11, 11,
      -10, -10, -6, -6, 6, 6, 10, 10, -11, -11, -4, -4, 4, 4, 11, 11, -12, 0, 0, 12,
      -12, -12, -1, -1, 1, 1, 12, 12, -9, -9, -8, -8, 8, 8, 9, 9, -11, -11, -5, -5,
      5, 5, 11, 11, -12, -12, -2, -2, 2, 2, 12, 12, -10, -10, -7, -7, 7, 7, 10, 10,
      -12, -12, -3, -3, 3, 3, 12, 12, -11, -11, -6, -6, 6, 6, 11, 11, -12, -12, -4, -4,
      4, 4, 12, 12, -9, -9, 9, 9, -10, -10, -8, -8, 8, 8, 10, 10, -13, -12, -12, -5,
      -5, 0, 0, 5, 5, 12, 12, 13, -13, -13, -11, -11, -7, -7, -1, -1, 1, 1, 7, 7,
      11, 11, 13, 13, -13, -13, -2, -2, 2, 2, 13, 13, -13, -13, -3, -3, 3, 3, 13, 13,
      -12, -12, -6, -6, 6, 6, 12, 12, -10, -10, -9, -9, 9, 9, 10, 10, -13, -13, -11,
      -11, -8, -8, -4, -4, 4, 4, 8, 8, 11, 11, 13, 13, -12, -12, -7, -7, 7, 7, 12, 12,
      -13, -13, -5, -5, 5, 5, 13, 13, -14, 0, 0, 14, -14, -14, -1, -1, 1, 1, 14, 14,
      -14, -14, -10, -10, -2, -2, 2, 2, 10, 10, 14, 14, -11, -11, -9, -9, 9, 9, 11, 11,
      -14, -14, -13, -13, -6, -6, -3, -3, 3, 3, 6, 6, 13, 13, 14, 14, -12, -12, -8, -8,
      8, 8, 12, 12, -14, -14, -4, -4, 4, 4, 14, 14, -13, -13, -7, -7, 7, 7, 13, 13, -11,
      -11, -10, -10, 10, 10, 11, 11, -14, -14, -5, -5, 5, 5, 14, 14, -15, -12, -12, -9,
      -9, 0, 0, 9, 9, 12, 12, 15, -15, -15, -1, -1, 1, 1, 15, 15, -15, -15, -2, -2,
      2, 2, 15, 15, -14, -14, -6, -6, 6, 6, 14, 14, -13, -13, -8, -8, 8, 8, 13, 13,
      -15, -15, -3, -3, 3, 3, 15, 15, -15, -15, -4, -4, 4, 4, 15, 15, -11, -11, 11, 11,
      -12, -12, -10, -10, 10, 10, 12, 12, -14, -14, -7, -7, 7, 7, 14, 14, -13, -13, -9, -9,
      9, 9, 13, 13, -15, -15, -5, -5, 5, 5, 15, 15, -16, 0, 0, 16, -16, -16, -1, -1,
      1, 1, 16, 16, -16, -16, -14, -14, -8, -8, -2, -2, 2, 2, 8, 8, 14, 14, 16, 16,
      -15, -15, -6, -6, 6, 6, 15, 15, -16, -16, -12, -12, -11, -11, -3, -3, 3, 3, 11, 11,
      12, 12, 16, 16, -13, -13, -10, -10, 10, 10, 13, 13, -16, -16, -4, -4, 4, 4, 16, 16,
      -15, -15, -7, -7, 7, 7, 15, 15, -14, -14, -9, -9, 9, 9, 14, 14, -16, -16, -5, -5,
      5, 5, 16, 16, -12, -12, 12, 12, -17, -15, -15, -8, -8, 0, 0, 8, 8, 15, 15, 17,
      -17, -17, -13, -13, -11, -11, -1, -1, 1, 1, 11, 11, 13, 13, 17, 17, -16, -16, -6, -6,
      6, 6, 16, 16, -17, -17, -2, -2, 2, 2, 17, 17, -14, -14, -10, -10, 10, 10, 14, 14,
      -17, -17, -3, -3, 3, 3, 17, 17, -17, -17, -16, -16, -7, -7, -4, -4, 4, 4, 7, 7,
      16, 16, 17, 17, -15, -15, -9, -9, 9, 9, 15, 15, -13, -13, -12, -12, 12, 12, 13, 13,
      -17, -17, -5, -5, 5, 5, 17, 17, -14, -14, -11, -11, 11, 11, 14, 14, -16, -16, -8]
xs = [0, 0, -1, 1, 0, -1, 1, -1, 1, 0, -2, 2, 0, -1, 1, -2, 2, -2, 2, -1, 1,
      -2, 2, -2, 2, 0, -3, 3, 0, -1, 1, -3, 3, -3, 3, -1, 1, -2, 2, -3, 3,
      -3, 3, -2, 2, 0, -4, 4, 0, -1, 1, -4, 4, -4, 4, -1, 1, -3, 3, -3, 3,
      -2, 2, -4, 4, -4, 4, -2, 2, 0, -3, 3, -4, 4, -5, 5, -4, 4, -3, 3, 0,
      -1, 1, -5, 5, -5, 5, -1, 1, -2, 2, -5, 5, -5, 5, -2, 2, -4, 4, -4, 4,
      -3, 3, -5, 5, -5, 5, -3, 3, 0, -6, 6, 0, -1, 1, -6, 6, -6, 6, -1, 1,
      -2, 2, -6, 6, -6, 6, -2, 2, -4, 4, -5, 5, -5, 5, -4, 4, -3, 3, -6, 6,
      -6, 6, -3, 3, 0, -7, 7, 0, -1, 1, -5, 5, -7, 7, -7, 7, -5, 5, -1, 1,
      -4, 4, -6, 6, -6, 6, -4, 4, -2, 2, -7, 7, -7, 7, -2, 2, -3, 3, -7, 7,
      -7, 7, -3, 3, -5, 5, -6, 6, -6, 6, -5, 5, 0, -8, 8, 0, -1, 1, -4, 4,
      -7, 7, -8, 8, -8, 8, -7, 7, -4, 4, -1, 1, -2, 2, -8, 8, -8, 8, -2, 2,
      -6, 6, -6, 6, -3, 3, -8, 8, -8, 8, -3, 3, -5, 5, -7, 7, -7, 7, -5, 5,
      -4, 4, -8, 8, -8, 8, -4, 4, 0, -9, 9, 0, -1, 1, -9, 9, -9, 9, -1, 1,
      -6, 6, -7, 7, -7, 7, -6, 6, -2, 2, -9, 9, -9, 9, -2, 2, -5, 5, -8, 8,
      -8, 8, -5, 5, -3, 3, -9, 9, -9, 9, -3, 3, -4, 4, -9, 9, -9, 9, -4, 4,
      -7, 7, -7, 7, 0, -6, 6, -8, 8, -10, 10, -8, 8, -6, 6, 0, -1, 1, -10, 10,
      -10, 10, -1, 1, -2, 2, -10, 10, -10, 10, -2, 2, -5, 5, -9, 9, -9, 9, -5, 5,
      -3, 3, -10, 10, -10, 10, -3, 3, -7, 7, -8, 8, -8, 8, -7, 7, -4, 4, -10, 10,
      -10, 10, -4, 4, -6, 6, -9, 9, -9, 9, -6, 6, 0, -11, 11, 0, -1, 1, -11, 11,
      -11, 11, -1, 1, -2, 2, -11, 11, -11, 11, -2, 2, -5, 5, -10, 10, -10, 10, -5, 5,
      -8, 8, -8, 8, -3, 3, -7, 7, -9, 9, -11, 11, -11, 11, -9, 9, -7, 7, -3, 3,
      -6, 6, -10, 10, -10, 10, -6, 6, -4, 4, -11, 11, -11, 11, -4, 4, 0, -12, 12, 0,
      -1, 1, -12, 12, -12, 12, -1, 1, -8, 8, -9, 9, -9, 9, -8, 8, -5, 5, -11, 11,
      -11, 11, -5, 5, -2, 2, -12, 12, -12, 12, -2, 2, -7, 7, -10, 10, -10, 10, -7, 7,
      -3, 3, -12, 12, -12, 12, -3, 3, -6, 6, -11, 11, -11, 11, -6, 6, -4, 4, -12, 12,
      -12, 12, -4, 4, -9, 9, -9, 9, -8, 8, -10, 10, -10, 10, -8, 8, 0, -5, 5, -12,
      12, -13, 13, -12, 12, -5, 5, 0, -1, 1, -7, 7, -11, 11, -13, 13, -13, 13, -11, 11,
      -7, 7, -1, 1, -2, 2, -13, 13, -13, 13, -2, 2, -3, 3, -13, 13, -13, 13, -3, 3,
      -6, 6, -12, 12, -12, 12, -6, 6, -9, 9, -10, 10, -10, 10, -9, 9, -4, 4, -8, 8,
      -11, 11, -13, 13, -13, 13, -11, 11, -8, 8, -4, 4, -7, 7, -12, 12, -12, 12, -7, 7,
      -5, 5, -13, 13, -13, 13, -5, 5, 0, -14, 14, 0, -1, 1, -14, 14, -14, 14, -1, 1,
      -2, 2, -10, 10, -14, 14, -14, 14, -10, 10, -2, 2, -9, 9, -11, 11, -11, 11, -9, 9,
      -3, 3, -6, 6, -13, 13, -14, 14, -14, 14, -13, 13, -6, 6, -3, 3, -8, 8, -12, 12,
      -12, 12, -8, 8, -4, 4, -14, 14, -14, 14, -4, 4, -7, 7, -13, 13, -13, 13, -7, 7,
      -10, 10, -11, 11, -11, 11, -10, 10, -5, 5, -14, 14, -14, 14, -5, 5, 0, -9, 9, -12,
      12, -15, 15, -12, 12, -9, 9, 0, -1, 1, -15, 15, -15, 15, -1, 1, -2, 2, -15, 15,
      -15, 15, -2, 2, -6, 6, -14, 14, -14, 14, -6, 6, -8, 8, -13, 13, -13, 13, -8, 8,
      -3, 3, -15, 15, -15, 15, -3, 3, -4, 4, -15, 15, -15, 15, -4, 4, -11, 11, -11, 11,
      -10, 10, -12, 12, -12, 12, -10, 10, -7, 7, -14, 14, -14, 14, -7, 7, -9, 9, -13, 13,
      -13, 13, -9, 9, -5, 5, -15, 15, -15, 15, -5, 5, 0, -16, 16, 0, -1, 1, -16, 16,
      -16, 16, -1, 1, -2, 2, -8, 8, -14, 14, -16, 16, -16, 16, -14, 14, -8, 8, -2, 2,
      -6, 6, -15, 15, -15, 15, -6, 6, -3, 3, -11, 11, -12, 12, -16, 16, -16, 16, -12, 12,
      -11, 11, -3, 3, -10, 10, -13, 13, -13, 13, -10, 10, -4, 4, -16, 16, -16, 16, -4, 4,
      -7, 7, -15, 15, -15, 15, -7, 7, -9, 9, -14, 14, -14, 14, -9, 9, -5, 5, -16, 16,
      -16, 16, -5, 5, -12, 12, -12, 12, 0, -8, 8, -15, 15, -17, 17, -15, 15, -8, 8, 0,
      -1, 1, -11, 11, -13, 13, -17, 17, -17, 17, -13, 13, -11, 11, -1, 1, -6, 6, -16, 16,
      -16, 16, -6, 6, -2, 2, -17, 17, -17, 17, -2, 2, -10, 10, -14, 14, -14, 14, -10, 10,
      -3, 3, -17, 17, -17, 17, -3, 3, -4, 4, -7, 7, -16, 16, -17, 17, -17, 17, -16, 16,
      -7, 7, -4, 4, -9, 9, -15, 15, -15, 15, -9, 9, -12, 12, -13, 13, -13, 13, -12, 12,
      -5, 5, -17, 17, -17, 17, -5, 5, -11, 11, -14, 14, -14, 14, -11, 11, -8, 8, -16]
circle_coords = np.empty((2, len(ys)), np.int32)
circle_coords[0, :] = np.array(ys)
circle_coords[1, :] = np.array(xs)
circle_coords = np.ascontiguousarray(circle_coords[:, :300])


def dark_lines(features):
    """
    Create the dark line mask with automatic grain boundary removal:

    - Use a difference of gaussians (DoG) to find thin dark lines, which are
      a guess for the location of decorated grain boundaries.
    - Replace grain boundaries with local maximums - this essentially
      removes them
    - Find high contrast local minimums
    - Apply a threshold to binarise, and clean up

    :param features: Dictionary of wafer features.
    :type features: dict
    :returns:  None

    """
    # Apply light smoothing
    im = features['im_normed']
    im_smoothed = features['im_smooth_0.5']

    # Difference of Gaussians
    dog = (cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=3) - im_smoothed)
    pixel_ops.BinaryThreshold(dog, 0.03)
    grain_bounrdary_guess = dog.astype(np.uint8)
    ys, xs = np.where(grain_bounrdary_guess)
    pixel_ops.FastThin(grain_bounrdary_guess, ys.copy(), xs.copy(), ip.thinning_lut)

    # Remove any candidates that don't have 2+ neighbours (end points and
    #  isolated pixels).
    count_mask = np.array([[1, 1, 1],
                           [1, 0, 1],
                           [1, 1, 1]], np.uint8)
    neighbour_count = cv2.filter2D(grain_bounrdary_guess, -1, count_mask,
                                   borderType=cv2.BORDER_REFLECT)
    pixel_ops.ApplyThresholdLT_U8_U8(neighbour_count, grain_bounrdary_guess, 2, 0)

    # find local maximums
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    local_maxs = cv2.dilate(im, se)

    # replace grain boundaries with local maximums
    im2 = im.copy()
    pixel_ops.CopyMaskF32(local_maxs, im2, grain_bounrdary_guess, 1)
    im2 = cv2.GaussianBlur(im2, ksize=(0, 0), sigmaX=0.75, borderType=cv2.BORDER_REPLICATE)

    if False:
        view = ImageViewer(im)
        view = ImageViewer(im2)
        view.show()
        sys.exit()

    # find local minimums
    red_defect = np.zeros(im.shape, np.uint8)
    pixel_ops.LocalMins(im2, red_defect)

    # band pass filter to refine the mask - only keep high contrast pixels
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    local_maxs = cv2.dilate(im2, se)
    local_contrast = (local_maxs - im2)
    pixel_ops.ApplyThresholdLT_F32_U8(local_contrast, red_defect, parameters.DARK_SENSITIVITY, 0)

    if False:
        print parameters.DARK_SENSITIVITY
        view = ImageViewer(red_defect)
        view.show()
        sys.exit()

    # thinning
    ys, xs = np.where(red_defect)
    pixel_ops.FastThin(red_defect, ys.copy(), xs.copy(), ip.thinning_lut)

    # remove small defects (likely noise)
    ccs, count = ip.connected_components(red_defect)
    grain_sizes = np.zeros(count + 1, np.int32)
    pixel_ops.CCSizes(ccs, grain_sizes)
    pixel_ops.RemoveSmallCCs(red_defect, ccs, grain_sizes, parameters.DARK_MIN_SIZE)
    features['mask_dark_lines'] = red_defect

    # metric to estimate the amount removed
    ridge_diff = (pixel_ops.CountEqual_U8(grain_bounrdary_guess, 1) -
                  pixel_ops.CountEqual_U8(red_defect, 1))
    features['ridges_removed'] = (ridge_diff / float(im.shape[0] * im.shape[1])) * 100

    if False:
        view = ImageViewer(im)
        view = ImageViewer(grain_bounrdary_guess)
        view = ImageViewer(features['mask_dark_lines'])
        view.show()
        sys.exit()

    return


def create_overlay(features, boundaries=None, dis_strong=False):
    impure = features['ov_impure2_u8'].astype(np.float32)
    foreground = features['ov_dislocations_u8'].astype(np.float32) / 255.
    orig = features['im_cropped_u8'].copy()

    rgb = np.empty((foreground.shape[0], foreground.shape[1], 3), np.uint8)

    # foreground
    b = orig + (foreground * 255)
    g = orig - 0.5 * (foreground * 255)
    r = orig - 0.5 * (foreground * 255)
    if dis_strong:
        # strong dislocations green, and the rest blue
        mask = foreground > parameters.STRONG_DIS_THRESHOLD
        b[mask] = 0
        g[mask] = 255
        r[mask] = 255

    # background
    if features['impure_area_fraction'] > 0:
        # show impure as red
        b -= impure
        g -= impure
        r += impure
    else:
        # show green line at boundary between impure and pure
        if features['impure_edge_width'] > 0:
            features['impure_edge_side'] = int(features['impure_edge_side'])
            if features['impure_edge_side'] in [0, 2]:
                pixels = int(round(features['impure_edge_width'] * orig.shape[0]))
            else:
                pixels = int(round(features['impure_edge_width'] * orig.shape[1]))
            if features['impure_edge_side'] == 0:
                r[pixels, :] = 255
                g[pixels, :] = 0
                b[pixels, :] = 0
            elif features['impure_edge_side'] == 2:
                r[-pixels, :] = 255
                g[-pixels, :] = 0
                b[-pixels, :] = 0
            elif features['impure_edge_side'] == 1:
                r[:, -pixels] = 255
                g[:, -pixels] = 0
                b[:, -pixels] = 0
            elif features['impure_edge_side'] == 3:
                r[:, pixels] = 255
                g[:, pixels] = 0
                b[:, pixels] = 0

    if 'mask_pinholes' in features:
        b[features['mask_pinholes']] = 255
        g[features['mask_pinholes']] = 0
        r[features['mask_pinholes']] = 255

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    if boundaries is not None:
        boundaries = boundaries.astype(np.float32) / 255.
        boundaries[boundaries < 0] = 0
        boundaries[boundaries > 1] = 1
        r = (r * (1 - boundaries)) + (np.zeros_like(r) * boundaries)
        b = (b * (1 - boundaries)) + (np.zeros_like(b) * boundaries)
        g = (g * (1 - boundaries)) + (np.ones_like(g) * 100 * boundaries)

    rgb[:, :, 0] = r.astype(np.uint8)
    rgb[:, :, 1] = g.astype(np.uint8)
    rgb[:, :, 2] = b.astype(np.uint8)

    return rgb


# @profile
def interpolate_background(im, dis_mask, features):
    """
    Given a wafer and a dislocation mask, interpolate masked areas
    with closest non-masked (hopefully pure) values.

    :param im: Normalised wafer image.
    :type im: ndimage
    :param dis_mask: Dislocation mask. 1=dislocation, 0=no dislocation
    :type dis_mask: ndimage
    :param features: Dictionary of wafer features.
    :type features: dict
    :returns:  Interpolated background image
    """
    global circle_coords

    # Downsize input (so the subsequent steps run faster)
    down_size = 2
    dis_mask_thumb = np.ascontiguousarray(dis_mask[::down_size, ::down_size])
    im_thumb = np.ascontiguousarray(im[::down_size, ::down_size])

    # Find the mean value of the pure areas. This will be used to
    # interpolate heavily dislocated areas
    pure_mean = pixel_ops.MaskMean_F32(im, dis_mask, 0)
    surface = np.ones(im_thumb.shape, np.float32) * pure_mean

    if False:
        view = ImageViewer(im_thumb)
        view = ImageViewer(dis_mask_thumb)
        view = ImageViewer(surface, vmin=im.min(), vmax=im.max())
        view.show()
        sys.exit()

    # Interpolate using an expanding spiral median filter that ignores masked
    #  pixels from the dislocation mask
    NUM_PIXELS = 100
    dislocations_filled = im_thumb.copy()
    scratch = np.empty((NUM_PIXELS), np.float32)
    pixel_ops.InterpolateDislocations(im_thumb, dis_mask_thumb, dislocations_filled,
                                      surface, scratch, circle_coords)

    if False:
        view = ImageViewer(im)
        ImageViewer(dis_mask_thumb)
        ImageViewer(dislocations_filled, vmin=im.min(), vmax=im.max())
        view.show()
        sys.exit()

    # Create full size background image from dislocations_filled
    background = im.copy()
    pixel_ops.FillMasks(background, dis_mask, dislocations_filled, down_size, im)

    # small amount of blurring
    background = cv2.GaussianBlur(background, ksize=(0, 0), sigmaX=5, borderType=cv2.BORDER_REPLICATE)

    if False:
        view = ImageViewer(im)
        view = ImageViewer(background, vmin=im.min(), vmax=im.max())
        view.show()
        sys.exit()

    return background


# @profile
def impurity_detection(features):
    """
    Create an impurity map with values in range [0, 1], with 0 meaning a very
    low probability of the corresponding pixel belonging to an impure area,
    and 1 meaning a high probability. The steps are as follows:

    - enhance ridges (brighter than surroundings) and trenches (darker than
      surroundings)
    - compute the ridge to trench ratio
    - use a sigmoid function to map the ratio values to [0, 1] probability values.
      A ratio above 1 means ridges are stronger, meaning the area is likely
      to have inverted ridge lines

    :param features: Dictionary of wafer features.
    :type features: dict
    :returns:  Impurity map
    """
    min_val = 0.00001
    im_smoothed = features['im_robust']
    h, w = im_smoothed.shape
    num_pixels = h * w
    h1, h2 = h // 4, (3 * h) // 4
    w1, w2 = w // 4, (3 * w) // 4

    if False:
        view = ImageViewer(im_smoothed)
        view.show()
        sys.exit()

    # Apply a filter to highlight ridges & trenches
    ridges = np.zeros(im_smoothed.shape, np.float32)
    pixel_ops.InvertedRidgeEnhance(im_smoothed, ridges)
    trenches = np.zeros(im_smoothed.shape, np.float32)
    pixel_ops.InvertedRidgeEnhance(-1 * im_smoothed, trenches)

    # Smooth, and then calculate the ridge-trench ratio
    ridge_smooth = ip.fast_smooth(ridges, sigma=25, pad_mode="reflect")
    trench_smooth = ip.fast_smooth(trenches, sigma=25, pad_mode="reflect")
    pixel_ops.ApplyThresholdLT_F32(trench_smooth, trench_smooth, min_val, min_val)

    # compute some global properties
    impure_ratio = (ridge_smooth / trench_smooth)
    features['rt_ratio_mean'] = cv2.mean(impure_ratio)[0]
    features['rt_ratio_middle_mean'] = impure_ratio[h // 3:(2 * h) // 3, w // 3:(2 * w) // 3].mean()
    features['rt_percentile_05'] = scipy.stats.scoreatpercentile(impure_ratio[::2, ::2].flat, per=5)

    if SAVE_FEATURES:
        features['f_1'] = 0
        features['f_2'] = 0
        features['f_3'] = 0
        # storage for impure classification
        # features['rt_min'] = impure_ratio.min()
        # features['rt_max'] = impure_ratio.max()
        # features['im_ridge'] = ridge_smooth
        # features['im_trench'] = trench_smooth

    if False:
        # print cv2.mean(ridges)[0]*1000
        print ridges.sum(), trenches.sum()
        view = ImageViewer(im_smoothed)
        ImageViewer(ridges, vmin=0, vmax=max(ridges.max(), trenches.max()))
        ImageViewer(trenches, vmin=0, vmax=max(ridges.max(), trenches.max()))
        ImageViewer(ridge_smooth, vmin=0, vmax=max(ridge_smooth.max(), trench_smooth.max()))
        ImageViewer(trench_smooth, vmin=0, vmax=max(ridge_smooth.max(), trench_smooth.max()))
        ImageViewer(impure_ratio, vmin=0, vmax=2)
        view.show()
        sys.exit()

    # sigmoid function
    OFFSET = -0.0001
    SLOPE = 0.75
    SIGMOID_SLOPE = 10000
    impure_prob = SLOPE + OFFSET + SLOPE * (ridge_smooth - 1) - trench_smooth
    impure_prob = 1.0 / (1.0 + np.exp(-impure_prob * SIGMOID_SLOPE))

    # compute area fraction with high impure probability - this will be useful
    #  for wafer classification. first for whole wafer, then for middle area.
    high_imp_count = pixel_ops.CountThresholdGT_F32(impure_prob, 0.5)
    features['inverted_ridge_area_fraction'] = (high_imp_count) / float(h * w)
    high_imp_count = pixel_ops.CountThresholdGT_F32(np.ascontiguousarray(impure_prob[h1:h2, w1:w2]), 0.5)
    features['inverted_ridge_middle'] = (100 * high_imp_count) / float((h2 - h1) * (w2 - w1))

    # analysis used to find impure edges. useful for impure areas with dense
    #  inverted ridges. this will be used to modify the background map
    def AnalyseProfile(profile, rt_ratio):
        if False:
            import matplotlib.pylab as plt
            plt.figure()
            plt.plot(profile)
            plt.show()
            # sys.exit()

        # find local minimums
        w4 = len(profile) // 4
        local_mins = np.logical_and(profile < np.roll(profile, 1), profile < np.roll(profile, -1))
        local_mins[0] = False
        local_mins[-1] = False
        local_mins = np.where(local_mins)[0]

        # left edge
        left_mins = local_mins[local_mins < w4]
        if len(left_mins) > 0:
            left_pos = left_mins[np.argmin(profile[left_mins])]
            left_min_pl = profile[left_pos]
            left_mean_rt = rt_ratio[:, :left_pos].mean()
            left_min_rest = profile[local_mins[local_mins > w4]].min()
        else:
            left_pos = -1
            left_min_pl = 0
            left_mean_rt = 0
            left_min_rest = 0

        # right_edge
        right_mins = local_mins[local_mins > 3 * w4]
        if len(right_mins) > 0:
            right_pos = right_mins[np.argmin(profile[right_mins])]
            right_min_pl = profile[right_pos]
            right_mean_rt = rt_ratio[:, right_pos:].mean()
            right_min_rest = profile[local_mins[local_mins < 3 * w4]].min()
        else:
            right_pos = -1
            right_min_pl = 0
            right_mean_rt = 0
            right_min_rest = 0

        background_pos = -1
        if left_min_pl < right_min_pl and left_min_pl < 0.75 and left_mean_rt > 1.0 and left_min_rest > 0.75:
            background_pos = left_pos
        elif right_min_pl < left_min_pl and right_min_pl < 0.75 and right_mean_rt > 1.0 and right_min_rest > 0.75:
            background_pos = right_pos

        return left_pos, left_min_pl, left_mean_rt, right_pos, right_min_pl, right_mean_rt, background_pos

    rows = im_smoothed[h1:h2, :].mean(axis=0)
    results_lr = AnalyseProfile(rows, impure_ratio)
    features['_background_pos_lr'] = results_lr[-1]
    cols = im_smoothed[:, w1:w2].mean(axis=1)
    results_tb = AnalyseProfile(cols, impure_ratio.T)
    features['_background_pos_tb'] = results_tb[-1]

    if False:
        # print top_val, bottom_val
        print features['left_1'], features['left_2'], features['left_3']
        print features['bottom_1'], features['bottom_2'], features['bottom_3']
        import matplotlib.pylab as plt
        plt.figure()
        plt.imshow(im_smoothed)
        plt.figure()
        plt.plot(im_smoothed[:, w1:w2].mean(axis=1), 'r')
        plt.plot(im_smoothed[h1:h2, :].mean(axis=0), 'g')
        plt.show()
        sys.exit()

    if False:
        print features['rt_ratio_middle_mean']
        view = ImageViewer(im_smoothed)
        view = ImageViewer(impure_ratio)
        ImageViewer(impure_prob, vmin=0, vmax=1)
        view.show()
        sys.exit()

    return impure_prob


# @profile
def create_dislocation_mask(im_flattened, features):
    """
    The input to this function is a wafer that:

    - areas with inverted grain boundaries have been replaced with local
      minimums and smoothed
    - the image has been normalised s.t. rows & cols all have an 80th
      percentile of 1. This largely removes edge/corner impurities

    The primary feature remaining in the image is dislocations, and the goal
    is to create a mask the covers them all, leaving only pure areas.

    The mask is created by the Cython function DislocationMask, which:

    - Finds pixels with a high gradient value
    - Drawing a short line from the pixel in direction of highest negative
      gradient. In other words, from bright to dark
    - This creates a mask that completely covers grain boundaries and
      small dislocations, and encapsulates larger dislocations

    The final step is to clean up the results:

    - Fill small holes in the mask using morphological operations
    - Fill larger holes by comparing the mean intensity of the hole with the
      surrounding area. Only fills dark holes. For example, a small fully pure
      grain may be enclosed by masked pixels, but we don't want to fill it.

    :param im: Normalised and corrected image of wafer.
    :type im: ndarray
    :param features: Dictionary of wafer features.
    :type features: dict
    :returns: The dislocation mask
    """
    # enhance edges
    smooth = cv2.GaussianBlur(im_flattened, (0, 0), 0.5, borderType=cv2.BORDER_REPLICATE)
    edgesH = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    edgesV = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    edges = cv2.magnitude(edgesH, edgesV)

    if False:
        view = ImageViewer(im_flattened)
        view = ImageViewer(edges)
        view.show()
        sys.exit()

    # create mask dislocations
    dis_mask = np.zeros(im_flattened.shape, np.uint8)
    coords = np.empty((2, 3), np.int32)
    pixel_ops.DislocationMask(smooth, edges, edgesH, edgesV, dis_mask,
                              parameters.MASK_SENSITIVITY, coords)

    if False:
        print "Sensitivity: ", parameters.MASK_SENSITIVITY
        assert dis_mask.shape == im_flattened.shape
        ip.trim_percentile(im_flattened, percentile=0.005)
        results = ip.overlay_mask(im_flattened, dis_mask)
        view = ImageViewer(im_flattened)
        view = ImageViewer(results)
        view = ImageViewer(dis_mask)
        view.show()
        sys.exit()

    # use morphology (closing) to clean up and fill small holes
    iterations = 1
    struct = ndimage.generate_binary_structure(2, 1).astype(np.uint8)
    c = np.ones((dis_mask.shape[0], 2), np.uint8)
    dis_mask = np.hstack((c, dis_mask, c))
    r = np.ones((2, dis_mask.shape[1]), np.uint8)
    dis_mask = np.vstack((r, dis_mask, r))
    dis_mask = cv2.morphologyEx(dis_mask, op=cv2.MORPH_CLOSE,
                                kernel=struct, iterations=iterations)
    dis_mask = np.ascontiguousarray(dis_mask[2:-2, 2:-2])

    if False:
        ip.trim_percentile(im_flattened, percentile=0.005)
        results = ip.overlay_mask(im_flattened, dis_mask)
        view = ImageViewer(results)
        view.show()
        sys.exit()

    # fill larger holes that are darker than surrounding
    num_pixels = im_flattened.shape[0] * im_flattened.shape[1]
    min_size = int(parameters.MIN_FILL_SIZE * num_pixels)
    max_size = int(parameters.MAX_FILL_SIZE * num_pixels)
    inverse = 1 - dis_mask
    ccs, count = ip.connected_components(inverse)

    if False:
        view = ImageViewer(im_flattened)
        view = ImageViewer(dis_mask)
        view = ImageViewer(inverse)
        view.show()
        sys.exit()

    # find average intensity in original image
    region_sizes = np.zeros(count + 1, np.int32)
    region_total_orig = np.zeros(count + 1, np.float64)
    pixel_ops.RegionProps(ccs, region_sizes, im_flattened, region_total_orig)
    region_sizes[region_sizes == 0] = 1
    region_mean_orig = region_total_orig / region_sizes

    # find average intensity in smoothed image
    region_sizes = np.zeros(count + 1, np.int32)
    region_total_smooth = np.zeros(count + 1, np.float64)
    smoothed2 = ip.fast_smooth(smooth, sigma=15)
    pixel_ops.RegionProps(ccs, region_sizes, smoothed2, region_total_smooth)
    region_sizes[region_sizes == 0] = 1
    region_mean_smooth = region_total_smooth / region_sizes
    pixel_ops.FilterRegions(ccs, inverse, region_sizes,
                            region_mean_orig, region_mean_smooth,
                            min_size, max_size, 0.05)
    dis_mask = 1 - inverse

    # warning if too much of the image is covered
    mask_coverage = (pixel_ops.CountEqual_U8(dis_mask, 1) /
                     float(dis_mask.shape[0] * dis_mask.shape[1]))
    if mask_coverage > 0.995:
        print "WARNING: The image is completely covered by the dislocation",
        print " mask, so it is being ignored. MASK_SENSITIVITY may be too low."
        dis_mask = np.zeros(im_flattened.shape, np.uint8)
    elif mask_coverage > 0.95:
        print r"WARNING: More than 95% of the image is covered by the",
        print r" dislocation mask. MASK_SENSITIVITY may be set too low."

    if False:
        ip.trim_percentile(im_flattened, percentile=0.005)
        results = ip.overlay_mask(im_flattened, dis_mask)
        view = ImageViewer(im_flattened)
        view = ImageViewer(results)
        view.show()
        sys.exit()

    return dis_mask


# @profile
def transition_correction(im):
    """
    The goal of this function is to create a transform that equalises impure
    edge and corner regions. This is achieved by "flattening" the image, by
    creating a transform that sets the 80th percentile of the rows and
    columns equal to 1. The transform is added to the input image to get the
    flattened imgae. The operation can be reverse by subtracting the
    transform.

    :param im: Normalised image of wafer.
    :type im: ndarray
    :returns:  The transformed input image and the correction
    """
    h, w = im.shape

    # columns - correct transitions along left and right
    col_per = np.apply_along_axis(scipy.stats.scoreatpercentile, 0, im, 80).astype(np.float32)
    col_smooth = ndimage.gaussian_filter1d(col_per, sigma=2, mode="nearest") - 1.0
    corrected = im - col_smooth.reshape((1, w))

    # rows - correct transitions along top and bottom
    row_per = np.apply_along_axis(scipy.stats.scoreatpercentile, 1, corrected, 80).astype(np.float32)
    row_smooth = ndimage.gaussian_filter1d(row_per, sigma=2, mode="nearest") - 1.0
    corrected -= row_smooth.reshape((h, 1))
    correction = corrected - im

    if False:
        view = ImageViewer(im)
        ImageViewer(corrected)
        ImageViewer(correction)
        view.show()
        sys.exit()

    return corrected, correction


# @profile
def wafer_classification(features):
    """
    Classify a wafer as a middle, edge, corner, transition or fully impure.
    The classification is rule based, and the main features are:

    - the presence of inverted ridge lines
    - various measures of the distribution of impure areas.

    :param features: Dictionary of wafer features.
    :type features: dict
    :returns:  None
    """
    if features["_downsized"] == 1:
        background = features['_background_reduced']
    else:
        background = features['_background']
    impure_area_fraction = features['impure_area_fraction']
    impure_mask = features['mask_impure_area']

    # precalculations
    # af = features['impure_area_fraction']
    num_pixels = (background.shape[0] * background.shape[1])
    num_impure = impure_area_fraction * num_pixels
    edge_width = int(math.ceil(impure_area_fraction * background.shape[0]))
    w = background.shape[0]
    w2 = w * w
    corner_width = int(round((-2 * w + math.sqrt(4 * w2 * (1 - impure_area_fraction))) / (-2)))

    # metrics based on background intensities
    s = 15
    s2 = s * 2
    a1, a2 = background[:s, :].mean(), background[s:s2, :].mean()
    b1, b2 = background[:, -s:].mean(), background[:, -s2:-s].mean()
    c1, c2 = background[-s:, :].mean(), background[-s2:-s, :].mean()
    d1, d2 = background[:, :s].mean(), background[:, s:s2].mean()
    e1, e2, e3, e4 = a2 - a1, b2 - b1, c2 - c1, d2 - d1
    edge_intensity = max(e1, e2, e3, e4)
    corner_intensity = max(min(e1, e2), min(e2, e3), min(e3, e4), min(e4, e1))

    # metrics based on distribution of impure areas in the impure mask
    middle_imp = 0
    edge_score = 0
    corner_score = 0
    avg_dist = 0
    edge_counts = np.zeros(4, np.float32)
    if num_impure > 1:
        middle_imp = (impure_mask[w // 4:(w * 3) // 4, w // 4:(w * 3) // 4].sum() / float((w // 2) ** 2))
        region_counts = np.zeros(8, np.float32)
        avg_dist = pixel_ops.ImpureRegionAnalysis(impure_mask, region_counts, edge_width + 1, corner_width + 1)
        region_counts /= num_impure
        avg_dist /= num_impure
        avg_dist = (avg_dist / float(w)) * 400
        corner_score = max(region_counts[4] + region_counts[5],
                           region_counts[5] + region_counts[6],
                           region_counts[6] + region_counts[7],
                           region_counts[7] + region_counts[4])

        edge_score = region_counts[:4].max()
        edge_counts = [impure_mask[:, 0].sum(), impure_mask[:, -1].sum(),
                       impure_mask[0, :].sum(), impure_mask[-1, :].sum()]
        edge_counts.sort()
        edge_counts = np.array(edge_counts, np.float32) / w

        if False:
            print region_counts
            print edge_counts
            view = ImageViewer(impure_mask)
            view.show()
            sys.exit()

    # rule-based classification
    if impure_area_fraction > 0.96:
        features['wafer_type'] = WaferType.FULLY_IMPURE
    else:
        if any([middle_imp > 0.01, edge_counts[1] > 0.45, features['rt_ratio_middle_mean'] > 1.27]):
            features['wafer_type'] = WaferType.TRANSITION
        else:
            if corner_intensity > 0.096 or features['corner_strength'] > 1.75:
                features['wafer_type'] = WaferType.CORNER
            else:
                if (edge_counts[2] > 0.3 and edge_counts[1] > 0.08) or (edge_counts[3] < 0.9 and avg_dist > 15):
                    features['wafer_type'] = WaferType.TRANSITION
                else:
                    if edge_counts[3] > 0.33 or (edge_counts[3] > 0.15 and features['edge_strength'] > 1.55):
                        features['wafer_type'] = WaferType.EDGE
                    else:
                        if features['rt_ratio_mean'] < 1.1 or features['defect_robust'] > 3:
                            features['wafer_type'] = WaferType.MIDDLE
                        else:
                            features['wafer_type'] = WaferType.TRANSITION

    if SAVE_FEATURES:
        # NOTE: do not delete, as this needs to be turned on for
        #  wafer_classification.py to work.
        features['wc_edge_impure'] = edge_score
        features['wc_edge_intensity'] = edge_intensity
        features['wc_corner_impure'] = corner_score
        features['wc_corner_intensity'] = corner_intensity
        features['wc_middle_imp'] = middle_imp
        features['wc_edge1'] = edge_counts[3]
        features['wc_edge2'] = edge_counts[2]
        features['wc_edge3'] = edge_counts[1]
        features['wc_edge4'] = edge_counts[0]

        # features['avg_dist'] = avg_dist

    features['impure_avg_edge_dist'] = avg_dist
    dx = features["impure_avg_edge_dist"]
    dy = (features["impure_area_fraction"] * 100) + 0.5
    features['edge_corner'] = np.rad2deg(np.arctan2(dy, dx))

    parameters.__WaferClassify(features)


# @profile
def feature_maps(features):
    """
    This is the high-level function that creates the feature maps: the
    foreground (defect) map, and the background (purity) map. Almost all
    of the effort goes into creating the interpolated background map,
    as the foreground is just the difference between the original image and the
    background.

    The steps are as follows:

    - impurity analysis
    - transition correction, to "flatten" images with impure edges and corners
    - create a mask that covers dislocations
    - interpolate the background map by infilling the dislocation mask
    - apply the inverse transition correction to the background map
    - foreground map = (original image - background map) * -1

    :param features: Dictionary of wafer features.
    :type features: dict
    :returns: None
    """
    gc.collect()

    im = features['im_saw_slope_corrected']
    im_smoothed = features['im_smooth_0.5']

    # Impurity analysis: find impure regions by detecting inverted ridges
    impure_map = impurity_detection(features)

    # replace impure areas with a smoothed version of
    #  the local mins so that the inverted ridges are mostly removed.
    #  Use the impure_map as a weighting factor to blend the local minimums
    #  with the original image
    local_mins = cv2.erode(im_smoothed, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10)))
    local_mins = ip.fast_smooth(local_mins, sigma=15, pad_mode="reflect")
    im_filtered = (impure_map * local_mins) + ((1.0 - impure_map) * im)

    if False:
        view = ImageViewer(im, vmin=0, vmax=1)
        ImageViewer(impure_map, vmin=0, vmax=1)
        ImageViewer(im_filtered, vmin=0, vmax=1)
        view.show()
        sys.exit()

    # - transition_correction: This doesn't actually identify edge/corner impure regions,
    #   but performs a normalisation that largely removes ones that
    #   are parallel to an edge (the most common case)
    im_flattened, correction = transition_correction(im_filtered)

    # create a dislocation mask
    dis_mask = create_dislocation_mask(im_flattened, features)

    if False:
        results = ip.overlay_mask(im, dis_mask)
        view = ImageViewer(im_flattened)
        ImageViewer(results)
        view.show()
        sys.exit()

    # interpolate the "background" map
    background = interpolate_background(im_flattened, dis_mask, features)

    # undo the transition corrections
    if correction is not None:
        background -= correction

        if False:
            view = ImageViewer(im)
            ImageViewer(background, vmin=im.min(), vmax=im.max())
            view.show()
            sys.exit()

    # correct impure edges that may be bright due to inverted ridges
    if features['_background_pos_lr'] > 0:
        minv = background[:, features['_background_pos_lr']].mean()
        if features['_background_pos_lr'] < background.shape[1] // 2:
            background[background[:, :features['_background_pos_lr']] > minv] = minv
        else:
            background[:, features['_background_pos_lr']:][background[:, features['_background_pos_lr']:] > minv] = minv
        # smooth
        a = max(0, features['_background_pos_lr'] - 10)
        b = min(background.shape[1], features['_background_pos_lr'] + 11)
        background[:, a:b] = cv2.GaussianBlur(np.ascontiguousarray(background[:, a:b]), (0, 0), 4,
                                              borderType=cv2.BORDER_REPLICATE)
    if features['_background_pos_tb'] > 0:
        minv = background[features['_background_pos_tb'], :].mean()
        if features['_background_pos_tb'] < background.shape[0] // 2:
            background[background[:features['_background_pos_tb'], :] > minv] = minv
        else:
            background[features['_background_pos_tb']:, :][background[features['_background_pos_tb']:, :] > minv] = minv
        # smooth
        a = max(0, features['_background_pos_tb'] - 10)
        b = min(background.shape[0], features['_background_pos_tb'] + 11)
        background[a:b, :] = cv2.GaussianBlur(np.ascontiguousarray(background[a:b, :]), (0, 0), 4,
                                              borderType=cv2.BORDER_REPLICATE)

    # early detection of fully impure (anything else should have some pure areas)
    features['bright_area_impure_prob'] = impure_map[im_smoothed > 0.9].mean()
    fi_line = [(1.25, 4), (0.95, 1.6)]
    fi = ((features['rt_ratio_middle_mean'] > 1.25) &
          ((((features['rt_ratio_middle_mean'] - fi_line[0][1]) * (fi_line[1][0] - fi_line[1][1])) -
            ((features['rt_percentile_05'] - fi_line[1][1]) * (fi_line[0][0] - fi_line[0][1]))) > 0))
    fi = fi or features['rt_ratio_middle_mean'] > 3.25
    if ((parameters.__FullyImpure(features) == True) or
            (parameters.__FullyImpure(features) is None and fi)):
        # fully impure
        lower = 0.3
        upper = 0.5
        background -= background.min()
        background *= (upper - lower) / background.max()
        background += lower

    if False:
        view = ImageViewer(im)
        ImageViewer(background)
        view.show()
        sys.exit()

    # create an impure area mask
    impure_area = np.zeros_like(background, dtype=np.uint8)
    pixel_ops.ApplyThresholdLT_F32_U8(background, impure_area, parameters.IMPURE_THRESHOLD, 1)
    num_pixels = float(im.shape[0] * im.shape[1])
    num_impure_pixels = cv2.sumElems(impure_area)[0]

    features['mask_impure_area'] = impure_area
    features['impure_area_fraction'] = num_impure_pixels / float(num_pixels)

    if features['impure_area_fraction'] > parameters.MIN_IMPURE_AREA:
        features['impure_area_mean_intensity'] = 1.0 - pixel_ops.MaskMean_F32(np.ascontiguousarray(im), impure_area, 1)
    else:
        features['impure_area_mean_intensity'] = 0
        features['impure_area_fraction'] = 0

    # new ratio metric
    impurePL = pixel_ops.MaskMean_F32(np.ascontiguousarray(im), impure_area, 1)
    normalPL = pixel_ops.MaskMean_F32(np.ascontiguousarray(im), impure_area, 0)
    if normalPL > 0:
        features['impure_area_mean_intensity_2'] = 1 - (impurePL/normalPL)
    else:
        features['impure_area_mean_intensity_2'] = 0

    # background histogram features
    background_bins = np.zeros((5), np.float32)
    pixel_ops.BackgrounHistogram(background, background_bins)
    background_bins = (background_bins / num_pixels) * 100
    features['background_hist_01'] = background_bins[0]
    features['background_hist_02'] = background_bins[1]
    features['background_hist_03'] = background_bins[2]
    features['background_hist_04'] = background_bins[3]
    features['background_hist_05'] = background_bins[4]

    # impure edge width
    def ImpEdges(imp_profile):
        pure_area = np.where(imp_profile > 0.5)[0]
        if len(pure_area) == 0: return 0, 0
        left = pure_area[0] / float(len(imp_profile))
        if left > 0.35: left = 0
        right = (len(imp_profile) - pure_area[-1]) / float(len(imp_profile))
        if right > 0.35: right = 0
        return left, right

    if features['impure_area_fraction'] > 0:
        imp_w_l, imp_w_r = ImpEdges(background.mean(axis=0))
        imp_w_t, imp_w_b = ImpEdges(background.mean(axis=1))
        edge_widths = [imp_w_t, imp_w_r, imp_w_b, imp_w_l]
        features['impure_edge_width'] = np.max(edge_widths)
        features['impure_edge_side'] = np.argmax(edge_widths)
    else:
        features['impure_edge_width'] = 0
        features['impure_edge_side'] = 0

    if False:
        print features['impure_edge_width'], features['impure_edge_side']
        view = ImageViewer(im)
        view = ImageViewer(background)
        view = ImageViewer(impure_area)
        view.show()
        sys.exit()

    # NOTE: We want the foreground map to be full-resolution
    if features["_downsized"] == 1:
        im = features['im_normed_full']
        features['_background_reduced'] = background
        h, w = im.shape
        background = cv2.resize(background, (w, h))

    # create "foreground" map by looking at the difference between the image
    #  and the background.
    foreground = background - im
    pixel_ops.ApplyThresholdLT_F32(foreground, foreground, 0, 0)
    pixel_ops.ApplyThresholdLT_F32(background, foreground, parameters.IMPURE_THRESHOLD, 0)
    features['_background'] = background
    features['ov_dislocations_u8'] = foreground

    if False:
        view = ImageViewer(im)
        ImageViewer(background)
        ImageViewer(foreground)
        view.show()
        sys.exit()

    return


# @profile
def edge_features(features):
    """
    Compute features to quantify the strength of impure edges.

    The following are the steps for computing edge_left:

    - select a horizontal strip across the middle 3rd of the image. This is
      done to exclude any impure regions along the top or bottom
    - compute the average intensity for each column
    - find the ratio: (max value in left region) / (value at left edge).
      This value will be high if the left edge is impure, since at the left
      edge the intensity will be low, but there will be a high value somewhere
      in the left region.

    The other edges are rotations of this process. The values for the
    individual edges are combined into higher-level features as follows:

    - edge_strength will be high if there is at least one impure edge
    - corner_strength will be high if there are two impure edges

    NOTE: Fully impure and transition wafers give unpredictable results for
    these features.

    :param features: Dictionary of wafer features.
    :type features: dict
    :returns: None
    """

    # compute left and right edge strengths based on column averages
    im = features['im_normed']
    h, w = im.shape

    if False:
        EDGE_THRESH = 0.66
        col_avg = ndimage.gaussian_filter1d(im[h // 3:(2 * h) // 3, :].mean(axis=0), sigma=2, mode="nearest")
        if col_avg[0] < EDGE_THRESH:
            edge_left = col_avg[:w // 3].max() / max(col_avg[0], 0.1)
        else:
            edge_left = 0
        if col_avg[-1] < EDGE_THRESH:
            edge_right = col_avg[(2 * w) // 3:].max() / max(col_avg[-1], 0.1)
        else:
            edge_right = 0

        # compute top and bottom edge strengths based on row averages
        row_avg = ndimage.gaussian_filter1d(im[:, w // 3:(2 * w) // 3].mean(axis=1), sigma=2, mode="nearest")
        if row_avg[0] < EDGE_THRESH:
            edge_top = row_avg[:h // 3].max() / max(row_avg[0], 0.1)
        else:
            edge_top = 0
        if row_avg[-1] < EDGE_THRESH:
            edge_bottom = row_avg[(2 * h) // 3:].max() / max(row_avg[-1], 0.1)
        else:
            edge_bottom = 0
    else:
        col_avg = ndimage.gaussian_filter1d(im[h // 3:(2 * h) // 3, :].mean(axis=0), sigma=2, mode="nearest")
        edge_left = col_avg[:w // 3].max() / max(col_avg[:w // 3].min(), 0.1)
        edge_right = col_avg[(2 * w) // 3:].max() / max(col_avg[(2 * w) // 3:].min(), 0.1)
        row_avg = ndimage.gaussian_filter1d(im[:, w // 3:(2 * w) // 3].mean(axis=1), sigma=2, mode="nearest")
        edge_top = row_avg[:h // 3].max() / max(row_avg[:h // 3].min(), 0.1)
        edge_bottom = row_avg[(2 * h) // 3:].max() / max(row_avg[(2 * h) // 3:].min(), 0.1)

    # combine individual edge stengths into edge/corner features
    features['edge_strength'] = max(edge_left, edge_right, edge_top, edge_bottom)
    features['corner_strength'] = min(max(edge_left, edge_right), max(edge_top, edge_bottom))

    if False:
        print edge_left, edge_right, edge_top, edge_bottom
        print features['edge_strength'], features['corner_strength']
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(col_avg)
        plt.plot(row_avg)
        plt.show()
        sys.exit()

    return


# @profile
def robust_dislocations(features):
    """
    Compute a robust dislocation metric that is, as far as possible, independent
     of the rest of the algorithm. The only information used from the main algorithm
     is the impure area.

    Compute two Difference of Gaussian images of the wafer. The first is a
     high-pass filter (to catch edges and small dislocations) and the second
     is a low-pass filter (to detect the middle of large dislocations). Add
     the two together and apply a threshold to get the defect mask.

    This has been tuned for both repeatability and predictive performance.

    :param features: Dictionary of wafer features.
    :type features: dict
    """

    defect_robust = 0
    defect_surface = 0

    smooth = features['im_robust']
    if features['impure_area_fraction'] < 0.90:
        # robust dislocation mask
        dog1 = (cv2.dilate(smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                             (parameters.DOG_STRUCT_SIZE,
                                                              parameters.DOG_STRUCT_SIZE))) - smooth)
        dog2 = (ip.fast_smooth(smooth, sigma=parameters.DOG_SIGMA2) -
                cv2.GaussianBlur(smooth, (0, 0), parameters.DOG_SIGMA1, borderType=cv2.BORDER_REPLICATE))
        dog = dog1 + dog2

        pixel_ops.ApplyThresholdGT_U8_F32(features['mask_impure_area'], dog, 0, 0)
        defect_mask = np.zeros_like(dog, np.uint8)
        pixel_ops.ApplyThresholdGT_F32_U8(dog, defect_mask, parameters.DOG_THRESH, 1)

        # cropping (edges are more likely to be unstable)
        c = parameters.ROBUST_CROP
        defect_mask = np.ascontiguousarray(defect_mask[c:-c, c:-c])
        impure = np.ascontiguousarray(features['mask_impure_area'][c:-c, c:-c])
        num_pure_pixels = pixel_ops.CountEqual_U8(impure, 0)

        # compute metric
        defect_robust = (pixel_ops.CountEqual_U8(defect_mask, 1) * 100.0) / num_pure_pixels

        # compute surface area
        eroded = defect_mask - cv2.erode(defect_mask, struct.astype(np.uint8))
        defect_pixels = float(pixel_ops.CountEqual_U8(defect_mask, 1))
        if defect_pixels > 0:
            defect_surface = pixel_ops.CountEqual_U8(eroded, 1) / defect_pixels
        else:
            defect_surface = 0

        if False:
            print defect_robust, defect_surface
            view = ImageViewer(features['im_normed'])
            ImageViewer(features['_background'])
            ImageViewer(dog1)
            ImageViewer(dog2)
            ImageViewer(dog)
            ImageViewer(defect_mask)
            view.show()
            sys.exit()

    # defect_robust -= parameters.ROBUST_METRIC_LOWER
    # defect_robust = max(0, defect_robust)
    defect_robust /= parameters.ROBUST_METRIC_UPPER
    defect_robust **= parameters.ROBUST_METRIC_POWER
    features['defect_robust'] = defect_robust * 10
    features['defect_surface'] = defect_surface

    return


# @profile
def dominant_orientation(features):
    """
    Determine if there is a strong, consistent orientation of wafer gradients.
     This my be caused by fully impure wafers with narrow and oriented grains
     emanating from the centre.

    :param features: Dictionary of wafer features.
    :type features: dict
    """

    im = features['im_robust']
    e = min(im.shape[0], im.shape[1]) // 4
    cropped = np.ascontiguousarray(im[e:-e, e:-e])

    # TODO: single step with cython??
    edgesH = cv2.Sobel(cropped, cv2.CV_32F, 0, 1)
    edgesV = cv2.Sobel(cropped, cv2.CV_32F, 1, 0)
    mag = cv2.magnitude(edgesH, edgesV)
    orientation = np.arctan2(edgesH, edgesV) * 2
    v = np.sin(orientation) * mag
    h = np.cos(orientation) * mag

    features['dominant_orientation'] = (math.sqrt(cv2.sumElems(h)[0] ** 2 + cv2.sumElems(v)[0] ** 2) * 1000 /
                                        (cropped.shape[0] * cropped.shape[1]))

    if False:
        print features['dominant_orientation']
        view = ImageViewer(im)
        view = ImageViewer(cropped)
        view = ImageViewer(mag)
        view = ImageViewer(orientation)
        view = ImageViewer(h)
        view = ImageViewer(v)
        view.show()
        sys.exit()


def pinholes(features):
    im = features['im_smooth_0.5']
    smooth = cv2.GaussianBlur(im, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    from skimage import draw

    pits = np.zeros_like(im)
    for R in [4, 6, 8, 10]:
        # for R in [5, 7, 9]:
        circle_rr, circle_cc = draw.circle_perimeter(R, R, R)
        w = (R * 2) + 1
        f = np.zeros((w, w), np.float32)
        f[circle_rr, circle_cc] = 1

        min_filtered = ndimage.minimum_filter(smooth, footprint=f)
        pits += min_filtered - smooth
    pits[pits < 0] = 0
    # pits[smooth > 0.35] = 0
    pit_mask = pits > 0.8
    s = ndimage.generate_binary_structure(2, 1)
    pit_mask = ndimage.binary_dilation(pit_mask, structure=s, iterations=3)

    features['mask_pinholes'] = pit_mask

    # compute some features
    ccs, num_ccs = ip.connected_components(pit_mask)
    if num_ccs > 0:
        with open("spot_features.txt", "a") as f:
            s = 5
            f.write("File: %s\n" % (os.path.split(features['filename'])[1]))
            for label in range(1, num_ccs + 1):
                spot_features = {}
                ys, xs = np.where(ccs == label)
                x1, x2 = xs.min() - s, xs.max() + s + 1
                y1, y2 = ys.min() - s, ys.max() + s + 1

                window = im[y1:y2, x1:x2]
                strength = 1.0 - window.min()
                y = ndimage.minimum_position(window)[0]  # + y1
                x = ndimage.minimum_position(window)[1]  # + x1
                profile = window[y:0:-1, x].copy()
                profile -= profile.min()
                profile /= profile.max()
                radius = np.where(profile > 0.8)[0][0]

                # spot_features['filename'] =
                # spot_features['location (x,y)'] = (x+x1, y+y1)
                # spot_features['strength'] = strength
                # spot_features['radius'] = radius
                f.write("""  %d.
    Location (x, y): (%d, %d)
    Strength: %0.02f
    Radius: %d
""" % (label, x + x1, y + y1, strength, radius))

                if False:
                    import matplotlib.pylab as plt
                    plt.figure()
                    plt.plot(profile)
                    # plt.show()
                    view = ImageViewer(window)
                    view.show()

    if False:
        view = ImageViewer(smooth)
        # view = ImageViewer(min_filtered)
        view = ImageViewer(pits)
        view = ImageViewer(pit_mask)
        view.show()
        # sys.exit()


def saw_mark_removal(im):
    h, w = im.shape
    col_middle = im[:, (w // 2) - 5:(w // 2) + 6].mean(axis=1)[20:-20]
    row_middle = im[(h // 2) - 5:(h // 2) + 6, :].mean(axis=0)[20:-20]
    col_middle -= ndimage.gaussian_filter1d(col_middle, sigma=1)
    row_middle -= ndimage.gaussian_filter1d(row_middle, sigma=1)
    col_var = col_middle[20:-20].std()
    row_var = row_middle[20:-20].std()

    # check if saw marks are horizonal or vertical
    flipped = False
    if col_var < row_var:
        im = im.T
        flipped = True

    # - determine period
    h, w = im.shape
    h2 = h // 2
    h8 = h // 8
    middle = im[h8:-h8, h2 - h8:h2 + h8]
    rows = middle.mean(axis=1)
    peaks = np.where((rows > np.roll(rows, 1)) &
                     (rows > np.roll(rows, -1)))[0]

    saw_mark_period = np.median(peaks - np.roll(peaks, 1))

    f1 = np.ones((int(saw_mark_period), 1), dtype=np.float32) / saw_mark_period
    no_saw = ndimage.convolve(im, f1, mode="reflect")

    if flipped:
        no_saw = no_saw.T

    if False:
        print "Saw mark period:", saw_mark_period
        import matplotlib.pylab as plt
        view = ImageViewer(im)
        ImageViewer(middle)
        plt.figure()
        plt.plot(rows)
        plt.plot(peaks, rows[peaks], 'o')
        ImageViewer(no_saw)
        view.show()

    return no_saw


def correct_slope(im):
    # We have a set from Hanwha that has a gradient across the wafer (cause unknown). It is leading to false
    #  impure areas and wafer type classifications. "Flatten" the middle of the wafer

    h, w = im.shape

    # correct columns
    cols = im.mean(axis=0)
    s1 = w//5
    s2 = w - s1
    vals = cols[s1:s2]
    xs = np.arange(s1, s2)
    rlm = sm.RLM(vals, sm.tools.add_constant(xs))
    model = rlm.fit()
    p0, p1 = model.params
    line_fit = (p0 + (p1*np.arange(w))).astype(np.float32)
    correction_cols = line_fit - line_fit.mean()
    correction_cols *= -1
    im_corrected = im + np.r_[correction_cols]

    if False:
        plt.figure()
        plt.plot(cols)
        plt.plot(line_fit)
        plt.plot(cols + correction_cols)
        view = ImageViewer(im)
        ImageViewer(im_corrected)
        view.show()
        sys.exit()

    # correct rows
    rows = im_corrected.mean(axis=1)
    s1 = h // 5
    s2 = h - s1
    vals = rows[s1:s2]
    xs = np.arange(s1, s2)
    rlm = sm.RLM(vals, sm.tools.add_constant(xs))
    model = rlm.fit()
    p0, p1 = model.params
    line_fit = (p0 + (p1 * np.arange(h))).astype(np.float32)
    correction_rows = line_fit - line_fit.mean()
    correction_rows *= -1
    im_corrected += np.c_[correction_rows]

    if False:
        plt.figure()
        plt.plot(rows)
        plt.plot(line_fit)
        plt.plot(rows + correction_rows)
        view = ImageViewer(im)
        ImageViewer(im_corrected)
        view.show()
        sys.exit()

    return im_corrected


# @profile
def feature_extraction(im, crop_props, fn=None, features=None):
    """
    A high-level function that calls other methods to compute metrics and
    create feature maps for PL image of wafer.

    :param im: Raw image of wafer.
    :type im: ndarray
    :param normalised_flag: True if the W2 is operating in lifetime mode. If
        so, parameters.BOUND_LOWER and parameters.BOUND_UPPER set the expected
        range of the normalised image intensities.
    :type normalised_flag: bool
    :returns: Dictionary of wafer features, including maps, masks and metrics.
    """
    # initialise features and add settings
    if features is None:
        features = {}
    features['crop_rotation'] = crop_props['estimated_rotation']
    features['param_alg_version'] = float(parameters.ver.replace('.', ''))
    if False:
        for p in dir(parameters):
            if p[0] == '_' or p in ['ver', 'HOST', 'SERVER_PORT', 'SET_NAME']: continue
            features['param_%s' % (p.lower())] = float(parameters.__getattribute__(p))
    features['wafer_type'] = WaferType.UNKNOWN
    if fn != None:
        features['filename'] = fn
    else:
        features['filename'] = 'N/A'

    # Optional preprocessing
    if not im.flags['C_CONTIGUOUS']:
        im = np.ascontiguousarray(im)
    if im.dtype.type is np.uint16:
        features['im_cropped_u16'] = im
    else:
        features['im_cropped_u16'] = im.astype(np.uint16)
    for _ in range(int(parameters.MEDIAN_FILTER_ITERATIONS)):
        im = cv2.medianBlur(im, ksize=3)
    im_pre = parameters.__Preprocess(im, features)
    if im_pre is not im:
        features['im_unprocessed'] = im.copy()
        im = im_pre

    # create the bit-layer mask. this is the cell-structure mask, and is included for completeness and consistency
    features['bl_cropped_u8'] = np.zeros(im.shape, np.uint8)

    # Feature extraction has been tuned and optimised for images that are
    #  approximately 500x500 pixels. Downsize high-resolution images so
    #  the features are computed correctly. NOTE: the foreground map will
    #  be computed using the full-resolution image
    if max(im.shape) > 700:
        features['im_full_size'] = im
        features["_downsized"] = 1
        h, w = im.shape
        im = cv2.resize(im, (w // 2, h // 2))
    else:
        features["_downsized"] = 0

    if False:
        im = ip.trim_percentile(im, 0.005)
        view = ImageViewer(im)
        view.show()
        sys.exit()

    # histogram based features - get information about the distribution of
    #  PL intensities. This is useful for image normalisation and some feature
    #  metrics.
    ip.histogram_percentiles(im, features)

    # Normalise image
    min_val = features['hist_percentile_01'] / float(features['hist_percentile_99.9'])
    norm_upper = features['hist_percentile_99.9']
    norm_lower = min(parameters.LOWER_CORRECTION, min_val)
    im_normed = ((im / norm_upper) - norm_lower) / (1 - norm_lower)
    features['im_normed'] = im_normed
    if features["_downsized"] == 1:
        features['im_normed_full'] = ((features['im_full_size'] / norm_upper) - norm_lower) / (1 - norm_lower)

    if 'input_param_skip_features' in features and int(features['input_param_skip_features']) == 1:
        return features

    # create a flat version of original image.
    flattened = im_normed.copy()
    if parameters.SAW_MARK_MULTI_WAFER:
        # saw mark removal
        flattened = saw_mark_removal(flattened)
        features['im_no_saw'] = flattened

    if parameters.SLOPE_MULTI_WAFER:
        flattened = correct_slope(flattened)
    features['im_saw_slope_corrected'] = flattened

    # pre-compute some features that are reused several times in subsequent processing
    # - a robust version of the smoothed image: removes salt & pepper/white
    #   noise and normalises based on all pixel values. should be consistent
    #   between captures of the same wafer
    # - lightly smoothed version of normalised image
    smooth = cv2.medianBlur(im, ksize=3)
    smooth = cv2.GaussianBlur(smooth, (0, 0), 1.0, borderType=cv2.BORDER_REPLICATE)
    smooth /= np.median(smooth[::2, ::2].flatten())
    features['im_robust'] = smooth
    features['im_smooth_0.5'] = cv2.GaussianBlur(im_normed, (0, 0), 0.5, borderType=cv2.BORDER_REPLICATE)

    if False:
        print parameters.LOWER_CORRECTION, norm_lower, min_val
        # view = ImageViewer(ip.trim_percentile(im, 0.005))
        view = ImageViewer(im_normed[140:250, 180:240], vmin=0.5, vmax=1)
        ImageViewer(features['im_robust'])
        view.show()
        sys.exit()

    if not parameters.FAST_MODE:
        # Determine the strength of the dominant orientation
        dominant_orientation(features)

    # feature maps. This also does wafer classification, since
    #  the wafer type is needed in order to create background map
    feature_maps(features)

    if not parameters.FAST_MODE:
        # "dark lines" features
        dark_lines(features)

        # a metric that measures the strength of decorated GBs
        decorated_gbs(features)

        # print "pinhole"
        # pinholes(features)

    # robust dislocation feature
    robust_dislocations(features)

    # Features that detect edges and corners
    edge_features(features)

    # wafer type analysis
    wafer_classification(features)

    if parameters.INCLUDE_SEMI:
        print "WARNING: SEMI features removed"

    return features


# @profile
def decorated_gbs(features):
    """
    Compute a metric to quantify the prevalance of decorated grain boundaries.
    This works by looking at the average line profile at high gradient locations,
    in the direction of greatest rate of change. If there are few dislocations
    and many grain boundaries, this will typically dip and recover quickly. On
    the other hand, if the high gradient location is at the edge of a dislocation,
    the PL count will drop and stay down.

    A similar metric looks at the line profile behaviour in the other direction
    (towards the pure area), giving an indication of how sparse/dense the
    dislocations are.

    :param features: Dictionary of wafer features.
    :type features: dict
    :returns: None
    """

    # find edges (areas with high gradient)
    im = features['im_normed']
    smooth = features['im_smooth_0.5']
    edgesH = cv2.Sobel(smooth, cv2.CV_32F, 0, 1)
    edgesV = cv2.Sobel(smooth, cv2.CV_32F, 1, 0)
    edges = cv2.magnitude(edgesH, edgesV)
    pixel_ops.ApplyThresholdGT_U8_F32(features['mask_impure_area'], edges, 0, 0)

    # compute the average line profile at the edges
    S = 10
    profile = np.zeros((S * 2) + 1, np.float32)
    count = pixel_ops.GradientProfile(im, edges, profile, 0.2)
    profile /= max(count, 1)

    # normalise
    features['decorated_gb_strength'] = profile[S + 4]
    features['defect_texture'] = profile[S - 5]
    features['decorated_gb_strength'] += 0.025
    features['decorated_gb_strength'] *= 100
    features['defect_texture'] -= 0.04
    features['defect_texture'] *= 100

    if False:
        print features['decorated_gb_strength']
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(profile)
        view = ImageViewer(smooth, vmin=0, vmax=1)
        view = ImageViewer(edges)
        view.show()
        sys.exit()

    return


# @profile
def combined_features(features):
    """
    This is the last function called when processing a wafer. It computes
    metrics and maps that require information from both W2 and G1.

    In theory, the G1 data may be used to update dislocation metrics. However,
     at the current time G1 isn't being used as few (no?) customers are using
     a G1 in production.

    :param features: PL results
    :type features: dict
    :param g1_results: G1 results (if available, otherwise None)
    :type g1_results: dict
    :returns: None
    """

    if 'ov_dislocations_u8' not in features:
        return

    # Compute metrics based on foreground map
    foreground = features['ov_dislocations_u8']

    # weak/strong dislocations
    num_foreground = foreground.shape[0] * foreground.shape[1]
    num_foreground_pure = int(round(num_foreground * (1.0 - features['impure_area_fraction'])))
    t = parameters.STRONG_DIS_THRESHOLD
    features['dislocations_weak'] = (pixel_ops.CountInRange_F32(foreground, 0.05, t) / float(num_foreground))
    features['dislocations_strong'] = (pixel_ops.CountInRange_F32(foreground, t, 1.0) / float(num_foreground))

    # Apply a threshold to the foreground map to get a dislocation mask
    dislocations = np.zeros_like(foreground, dtype=np.uint8)
    pixel_ops.ApplyThresholdGT_F32_U8(foreground, dislocations, parameters.DISLOCATION_THRESHOLD, 1)
    features['dislocation_mask_whole_wafer'] = cv2.sumElems(dislocations)[0] / float(num_foreground)
    if features['impure_area_fraction'] < 1.0:
        pure_count = pixel_ops.CountGT_F32_U8(features['_background'], dislocations, 1, parameters.IMPURE_THRESHOLD)
        features['dislocation_mask_pure_area'] = pure_count / float(num_foreground_pure)
    else:
        features['dislocation_mask_pure_area'] = 0

    if False:
        print parameters.DISLOCATION_THRESHOLD, features['dislocation_mask_pure_area']
        view = ImageViewer(features['im_normed'])
        view = ImageViewer(dislocations)
        view.show()
        sys.exit()

    if not parameters.FAST_MODE:
        # average intensity of a dislocation
        pure_dislocation_mask = ((features['_background'] > parameters.IMPURE_THRESHOLD) & (dislocations == 1))
        if pure_dislocation_mask.sum() > 0:
            features['dislocation_area_mean_intensity'] = features['im_cropped_u16'][pure_dislocation_mask].mean()
        else:
            features['dislocation_area_mean_intensity'] = 0.0

        # Features based on the dark lines mask
        dark_lines = features['mask_dark_lines']
        num_pixels = dark_lines.shape[0] * dark_lines.shape[1]
        num_pure_pixels = int(round(num_pixels * (1.0 - features['impure_area_fraction'])))
        dark_lines_count = cv2.sumElems(dark_lines)[0]
        features['dark_lines_whole_wafer'] = dark_lines_count / float(num_pixels)
        if num_pure_pixels > 0:
            dark_count = pixel_ops.CountInMaskEqual_U8(features['mask_impure_area'], dark_lines, 1, 0)
            features['dark_lines_pure_area'] = dark_count / float(num_pure_pixels)
        else:
            features['dark_lines_pure_area'] = 0

    if False:
        # print parameters.IMPURE_THRESHOLD, features['dark_lines_pure_area']
        view = ImageViewer(features['_background'])
        ImageViewer(features['ov_dislocations_u8'])
        ImageViewer(dislocations)
        view.show()
        sys.exit()

    # prepare images for display and convert to 8 bit
    if features["_downsized"] == 1:
        normed = features['im_normed_full']
    else:
        normed = features['im_normed']
    pixel_ops.ClipImage(np.ascontiguousarray(normed), 0.0, 1.0)
    #features['im_cropped_u8'] = (normed * 255).astype(np.uint8)

    # changed type: remove dark pixels in overlay images.
    features['im_cropped_u8'] = (normed * 255).astype(np.float32)

    if False:
        view = ImageViewer(normed)
        ImageViewer(features['im_cropped_u8'])
        view.show()
        sys.exit()

    imp_cutoff = parameters.IMPURE_THRESHOLD
    impure = features['_background'].copy()
    pixel_ops.ApplyThresholdGT_F32(impure, impure, imp_cutoff, imp_cutoff)
    impure /= imp_cutoff
    impure = 1 - impure
    pixel_ops.ClipImage(impure, 0, 1)
    features['ov_impure2_u8'] = (impure * 255).astype(np.uint8)

    pixel_ops.ClipImage(features['ov_dislocations_u8'], 0, 1)
    features['ov_dislocations_u8'] = (features['ov_dislocations_u8'] * 255).astype(np.uint8)

    return


def main():
    pass


if __name__ == '__main__':
    main()
