import numpy as np
from image_processing import ImageViewer
import image_processing as ip
import cv2
from scipy import ndimage, stats, optimize
import sys
import parameters
import pixel_ops
import cropping
import timeit
import matplotlib.pylab as plt
import cell_processing as cell
import features_multi_wafer as multi_wafer
import features_cz_cell as cz_cell
import numpy.ma as ma
from scipy.spatial import distance
import math
import collections


def create_overlay(features):
    h, w = features['im_cropped_u8'].shape

    if True and 'ov_impure2_u8' in features:
        impure = features['ov_impure2_u8']
    else:
        impure = np.zeros_like(features['im_cropped_u8'], np.uint8)

    if 'ov_dislocations_u8' in features:
        foreground = features['ov_dislocations_u8']  # .astype(np.float32) / 255.
    else:
        foreground = np.zeros_like(features['im_cropped_u8'], np.uint8)

    orig = features['im_cropped_u8'].astype(np.int32)

    if True:
        # foreground
        b = orig + foreground
        g = orig - foreground
        r = orig - foreground
    else:
        b = orig.copy()
        g = orig.copy()
        r = orig.copy()

    if True:
        # splotches
        if "ov_splotches_u8" in features:
            splotches = features["ov_splotches_u8"]  # * 2
            b += splotches
            g -= splotches
            r -= splotches

    if True:
        # show impure as red
        b -= impure
        g -= impure
        r += impure

    if False:
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

    # bright lines
    if True and "ov_bright_lines_u8" in features:
        broken_fingers = features["ov_bright_lines_u8"] * 2
        r -= broken_fingers
        g += broken_fingers
        b -= broken_fingers

    if "ov_bright_area_u8" in features:
        broken_fingers = features["ov_bright_area_u8"] * 3
        r -= broken_fingers
        g += broken_fingers
        b -= broken_fingers

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:, :, 0] = r.astype(np.uint8)
    rgb[:, :, 1] = g.astype(np.uint8)
    rgb[:, :, 2] = b.astype(np.uint8)

    # cracks
    if "mk_cracks_u8" in features:
        im_rgb = ip.overlay_mask(rgb, features['mk_cracks_u8'], 'r')

    # broken fingers
    if True and "mk_finger_break_u8" in features:
        rgb = ip.overlay_mask(rgb, features['mk_finger_break_u8'], 'b')

    return rgb


def between_bb_mono(im, bb_locs):
    mono_lr = np.empty_like(im)
    mono_rl = np.empty_like(im)
    pixel_ops.MakeMonotonicBBs(im, mono_lr, bb_locs)
    pixel_ops.MakeMonotonicBBs(np.ascontiguousarray(im[:, ::-1]), mono_rl,
                               np.ascontiguousarray(im.shape[1] - bb_locs[::-1]))
    mono_rl = mono_rl[:, ::-1]
    return np.minimum(mono_lr, mono_rl)


def interpolate_background(im, dis_mask):
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
                                      surface, scratch, multi_wafer.circle_coords)

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
    background = cv2.GaussianBlur(background, ksize=(0, 0), sigmaX=1, borderType=cv2.BORDER_REPLICATE)

    if False:
        view = ImageViewer(im)
        view = ImageViewer(background, vmin=im.min(), vmax=im.max())
        view.show()
        sys.exit()

    return background


def line_error(params, x_locs, profile):
    # fit a line that is closer to peaks (pure areas) than troughs (likely dislocations)
    line_vals = x_locs * params[0] + params[1]
    errors = profile - line_vals

    if True:
        errors[np.sign(errors) > 0] *= 3

    return np.abs(errors).mean()


# @profile
def efficiency_analysis(features):
    im = features['im_no_fingers']
    im_peaks = features['im_norm'][features['_peak_row_nums'], :]
    bbs = features['_busbar_cols']

    # make sure no zero values
    im_peaks = im_peaks.copy()
    pixel_ops.ApplyThresholdLT_F32(im_peaks, im_peaks, 0.01, 0.01)
    im = im.copy()
    pixel_ops.ApplyThresholdLT_F32(im, im, 0.01, 0.01)

    # IDEA:
    # two defect mask:
    # 1. existing one (highlights anything dark)
    # 2. do a dark line (local mins). mask out & interpolate.
    #    this one won't pick up grain boundaries
    # - slider blends between them

    #################
    # DEFECT MASK 1 #
    #################
    # - very sensitive to all dark areas

    # need to interpolate values along edges/busbars so defects in these regions can be found
    xs = np.arange(im_peaks.shape[0])
    cols = [features['cell_edge_left'], features['cell_edge_right']]
    cols += list(np.where(features['mask_busbar_edges'][0, :])[0])
    for c in cols:
        ys = im_peaks[:, c].copy()
        ys[ys > features['_bright_area_thresh']] = features['_bright_area_thresh']
        params_op = optimize.fmin_powell(line_error, [0.0, ys.mean()], disp=0, args=(xs, ys), xtol=0.05, ftol=0.05)
        vals = xs * params_op[0] + params_op[1]

        im_peaks[:, c] = vals

        if False:
            print features['_bright_area_thresh']
            mask = np.zeros_like(im, np.uint8)
            mask[:, c] = 1
            ImageViewer(ip.overlay_mask(im, mask))
            plt.figure()
            plt.plot(ys)
            plt.plot(vals)
            plt.show()

    # max of actual val and: interpolate vertical line at:
    # - left & right edge
    # - left & right of each BB

    # make monotonic (don't care about local mins)
    bb_mono = between_bb_mono(im_peaks, bbs)
    background1 = bb_mono
    background1 = cv2.GaussianBlur(background1, ksize=(0, 0), sigmaX=3, borderType=cv2.BORDER_REPLICATE)

    # relative difference
    foreground1 = background1 / im_peaks
    foreground1 -= 1
    pixel_ops.ClipImage(foreground1, 0, 3.0)
    foreground1[:, features['mask_busbar_filled'][0, :]] = 0

    # expand to full size
    full_size = np.zeros(im.shape, np.float32)
    pixel_ops.ExpandFingers(full_size, foreground1, features['_peak_row_nums'])
    foreground1 = full_size
    background_full = np.zeros(im.shape, np.float32)
    pixel_ops.ExpandFingers(background_full, background1, features['_peak_row_nums'])

    if False:
        view = ImageViewer(im)
        ImageViewer(foreground1)
        view.show()

    #################
    # DEFECT MASK 2 #
    #################
    # - less likely to include dark grains
    # - mask out local mins and areas of high gradient, and the interpolate background
    flatten_cols = np.r_[im.mean(axis=0)]
    im_flattened = im - flatten_cols
    flatten_rows = np.c_[im_flattened.mean(axis=1)]
    im_flattened -= flatten_rows
    im_flattened[features['mask_busbar_filled']] = 0
    im_smoothed = cv2.GaussianBlur(im_flattened, ksize=(0, 0), sigmaX=2)

    # local mins
    dark_lines = np.zeros_like(im_flattened, np.uint8)
    pixel_ops.LocalMins(im_smoothed, dark_lines)
    s = ndimage.generate_binary_structure(2, 1)
    dark_lines = ndimage.binary_dilation(dark_lines, structure=s)

    # high gradient
    edgesH = cv2.Sobel(im_smoothed, cv2.CV_32F, 1, 0)
    edgesV = cv2.Sobel(im_smoothed, cv2.CV_32F, 0, 1)
    edges = cv2.magnitude(edgesH, edgesV)

    # combine
    defect_candidates = ((edges > 0.2) | dark_lines)

    if False:
        view = ImageViewer(im_flattened)
        ImageViewer(ip.overlay_mask(im_flattened, dark_lines))
        ImageViewer(ip.overlay_mask(im_flattened, edges > 0.2))
        ImageViewer(ip.overlay_mask(im_flattened, defect_candidates))
        view.show()
        sys.exit()

    background2 = interpolate_background(im_flattened, defect_candidates.astype(np.uint8))
    background2 += flatten_rows
    background2 += flatten_cols

    # relative difference
    foreground2 = background2 / im
    foreground2 -= 1
    pixel_ops.ClipImage(foreground2, 0, 3.0)
    foreground2[features['mask_busbar_filled']] = 0

    if False:
        view = ImageViewer(im)
        ImageViewer(background2)
        ImageViewer(foreground2)
        view.show()
        sys.exit()

    if False:
        features['_defect1'] = foreground1
        features['_defect2'] = foreground2

    # discrimination function
    x1, y1 = 0.0, 0.5
    x2, y2 = 0.4, 0.0
    foreground = (-1 * ((y2 - y1) * foreground2 - (x2 - x1) * foreground1 + x2 * y1 - y2 * x1) /
                  np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2))

    foreground += 0.1 + parameters.CELL_DISLOCATION_SENSITIVITY
    foreground *= 0.25
    pixel_ops.ClipImage(foreground, 0, 1)

    features['ov_dislocations_u8'] = (foreground * 255).astype(np.uint8)

    if False:
        view = ImageViewer(im)
        # ImageViewer(foreground1)
        # ImageViewer(foreground2)
        ImageViewer(foreground)
        ImageViewer(background_full)
        # ImageViewer(foreground > 0.2)
        view.show()
        sys.exit()

    ################
    # IMPURE AREAS #
    ################

    def FindImpure(imp_profile, imp_profile_r, defects, debug=False):
        if False:
            ImageViewer(im)
            plt.figure()
            plt.plot(imp_profile)
            plt.plot(defects)
            plt.show()

        imp_ratio = (imp_profile / imp_profile_r).astype(np.float32)
        imp_ratio[imp_ratio > 0.9] = 0.9
        imp_ratio /= 0.9
        global_min = np.argmin(imp_ratio)
        local_mins = np.where((imp_ratio < np.roll(imp_ratio, 1)) & (imp_ratio < np.roll(imp_ratio, -1)))[0]

        # make monotonic from global minimum
        imp_mono = imp_ratio.copy()
        flip = False
        if global_min > len(imp_mono) // 2:
            flip = True
            imp_mono = imp_mono[::-1]
            global_min = np.argmin(imp_mono)
        rest = np.ascontiguousarray(imp_mono[global_min:])
        rest_mono = np.empty_like(rest)
        pixel_ops.MakeMonotonic(rest, rest_mono)
        imp_mono[global_min:] = rest_mono
        # if edge_dist < 0.075:
        #     imp_mono[:global_min] = imp_ratio[global_min]
        if flip:
            imp_mono = imp_mono[::-1]
            global_min = np.argmin(imp_mono)

        # for a real impure area, the global minimum should be close to an edge
        edge_dist = min(global_min, len(imp_ratio) - global_min) / float(len(imp_ratio))
        edge_spot = int(0.03 * len(imp_mono))
        imp_at_edge = min(imp_mono[edge_spot], imp_mono[-edge_spot])
        imp_1d = np.ones_like(imp_mono)

        # check the defect content in the "impure" area
        if (imp_mono < 1.0).sum() == 0:
            defect_strength = 0
        else:
            # defect_strength = defects[imp_mono < 1.0].mean()
            defect_strength = np.median(defects[imp_mono < 1.0])

        if (edge_dist < 0.075 and  # lowest point close to edge
                    imp_at_edge < 0.8 and  # edge is dark
                    defect_strength < 0.1 and  # not too many defects
            # (global_min == local_mins[0] or global_min == local_mins[-1]) and # no local mins closer to edge
                    (imp_mono - imp_ratio).mean() < 0.035):  # no large non-impure dips
            imp_1d = imp_mono.copy()
            imp_1d -= 0.3
            if global_min < len(imp_profile) // 2:
                imp_1d[:global_min] = imp_1d[global_min]
            else:
                imp_1d[global_min:] = imp_1d[global_min]

        if debug:
            print edge_dist, edge_dist < 0.075
            print imp_at_edge, imp_at_edge < 0.8
            print defect_strength, defect_strength < 0.1
            # print global_min, local_mins[0], (global_min == local_mins[0] or global_min == local_mins[-1])
            print (imp_mono - imp_ratio).mean(), (imp_mono - imp_ratio).mean() < 0.035
            ImageViewer(im)
            plt.figure()
            plt.plot(imp_profile, 'r')
            plt.plot(imp_mono, 'g')
            plt.plot(imp_ratio, 'b')
            # plt.plot((0, len(signal_var)), (0.1, 0.1))
            plt.plot(defects, 'c')
            plt.plot(imp_1d, 'c')

            plt.show()

        # find impure edge width
        THRESH = 0.5
        if imp_1d[0] < THRESH:
            edge_width = np.where(imp_1d < THRESH)[0][-1] / float(len(imp_profile))
            left = True
        elif imp_1d[-1] < THRESH:
            edge_width = (len(imp_profile) - np.where(imp_1d < THRESH)[0][0]) / float(len(imp_profile))
            left = False
        else:
            edge_width = 0
            left = True

        return imp_1d, edge_width, left

    if False:
        # ignore defect areas
        defect_mask = foreground > 0.2
        mx = ma.masked_array(im, mask=defect_mask)

        if False:
            view = ImageViewer(im)
            ImageViewer(defect_mask)
            view.show()

    # left/right edges
    cols = np.apply_along_axis(stats.scoreatpercentile, 0, im, per=75)
    cols[cols > 1.0] = 1.0
    imp_profile = ndimage.gaussian_filter1d(cols, sigma=3, mode="nearest")
    imp_profile_r = imp_profile[::-1]
    if len(bbs) >= 2:
        mid = (bbs[0] + bbs[-1]) // 2
    else:
        mid = im.shape[1] // 2
    imp_profile_r = np.roll(imp_profile_r, mid - (len(imp_profile) // 2))
    col_defect = foreground.mean(axis=0)
    imp_h, ew_h, e_left = FindImpure(imp_profile, imp_profile_r, col_defect, debug=False)

    # top/bottom edges
    rows = np.apply_along_axis(stats.scoreatpercentile, 1, im, per=75)
    rows[rows > 1.0] = 1.0
    imp_profile = ndimage.gaussian_filter1d(rows, sigma=3, mode="nearest")
    imp_profile_r = imp_profile[::-1]
    row_defect = foreground.mean(axis=1)
    imp_v, ew_v, e_top = FindImpure(imp_profile, imp_profile_r, row_defect, debug=False)

    if ew_h > ew_v:
        features['impure_edge_width'] = ew_h
        if e_left:
            features['impure_edge_side'] = 3
        else:
            features['impure_edge_side'] = 1
    else:
        features['impure_edge_width'] = ew_v
        if e_top:
            features['impure_edge_side'] = 0
        else:
            features['impure_edge_side'] = 2

    # combine
    impure_h = np.ones_like(im)
    impure_h[:, :] = np.r_[imp_h]
    impure_v = np.ones_like(im)
    impure_v[:, :] = np.c_[imp_v]
    impure = np.minimum(impure_h, impure_v)

    if False:
        print features['impure_edge_width'], features['impure_edge_side']
        view = ImageViewer(impure_h)
        ImageViewer(impure_v)
        ImageViewer(impure)
        view.show()
        sys.exit()

    imp_cutoff = 0.55
    pixel_ops.ApplyThresholdGT_F32(impure, impure, imp_cutoff, imp_cutoff)
    impure /= imp_cutoff

    impure_overlay = np.log10(2 - impure)

    if False:
        plt.figure()
        plt.plot(impure[:, 0], 'r')
        plt.plot(np.log(impure[:, 0] + 1), 'g')
        plt.plot(np.log10(impure[:, 0] + 1), 'b')
        plt.show()

        view = ImageViewer(impure)
        view.show()

    pixel_ops.ClipImage(impure_overlay, 0, 1)
    features['ov_impure2_u8'] = (impure_overlay * 255).astype(np.uint8)

    ###########
    # METRICS #
    ###########
    num_pixels = im.shape[0] * im.shape[1]

    # impure
    impure_thresh = 0.9  # 0.8
    dislocation_thresh = 0.1
    num_impure_pixels = pixel_ops.CountThresholdLT_F32(impure, impure_thresh)
    features['impure_area_fraction'] = (num_impure_pixels / float(num_pixels))

    if features['impure_area_fraction'] > 0.001:
        pure_mask = (impure > impure_thresh) & (foreground < dislocation_thresh)
        features['impure_strength'] = (im[pure_mask].mean() / im[impure < impure_thresh].mean()) - 1
    else:
        features['impure_strength'] = 0
        features['impure_area_fraction'] = 0

    features['impure_strength2'] = features['impure_strength'] * features['impure_area_fraction'] * 100

    # dislocations
    num_dislocations = pixel_ops.CountThresholdGT_F32(foreground, dislocation_thresh)
    features['dislocation_area_fraction'] = (num_dislocations / float(num_pixels))
    features['_foreground'] = foreground
    features['_dislocation_thresh'] = dislocation_thresh
    features['_impure'] = impure
    features['_impure_thresh'] = impure_thresh

    # find the average distance between dislocation pixels
    ys, xs = np.where(foreground > dislocation_thresh)
    points = np.c_[ys, xs]
    num_samples = 1500
    if len(ys) > 0:
        points = points[::max(1, points.shape[0] // num_samples), :]
        pixel_dists = distance.pdist(points)
        avg_dist = np.mean(pixel_dists)
        avg_dist = (avg_dist - 0.3 * im.shape[0]) / (0.5 * im.shape[0])
    else:
        avg_dist = 0
    features['dislocation_density'] = 1.0 - avg_dist

    if len(ys) > 0:
        dis_vals = foreground[ys, xs]
        dislocation_strength = math.sqrt((dis_vals ** 2).mean())
        features['dislocation_severity_A'] = dislocation_strength

        # create a second defect strength.
        ratio2 = background_full / im
        ratio2 = np.clip(ratio2, 1, 5)
        ratio2[:, features['mask_busbar_filled'][0, :]] = 1
        features['dislocation_severity_B'] = ratio2[ys, xs].mean()

        if False:
            view = ImageViewer(background_full)
            ImageViewer(im)
            ImageViewer(ratio2)
            view.show()
    else:
        features['dislocation_severity_A'] = 0
        features['dislocation_severity_B'] = 0

    if False:
        view = ImageViewer(im)
        foreground[foreground < 0] = 0
        ImageViewer(foreground)
        ImageViewer(background_full)
        view.show()
        sys.exit()

    # dislocation histogram features
    foreground_bins = np.zeros((5), np.float32)
    pixel_ops.BackgrounHistogram(foreground, foreground_bins)
    foreground_bins = (foreground_bins / num_pixels) * 100
    features['dislocation_hist_01'] = foreground_bins[0]
    features['dislocation_hist_02'] = foreground_bins[1]
    features['dislocation_hist_03'] = foreground_bins[2]
    features['dislocation_hist_04'] = foreground_bins[3]
    features['dislocation_hist_05'] = foreground_bins[4]
    features['dislocation_strength'] = features['dislocation_area_fraction'] * features['dislocation_severity_A']

    if False:
        # print features['foreground_hist_01'], features['foreground_hist_02'], features['foreground_hist_03'],
        # print features['foreground_hist_04'], features['foreground_hist_05']
        view = ImageViewer(im)
        ImageViewer(foreground)
        ImageViewer(impure)
        ImageViewer(create_overlay(features))
        view.show()


def bright_areas(features):
    im = features['im_norm']
    row_nums = features['_finger_row_nums']
    im_fingers = features['im_norm'][row_nums, :]

    vals = np.sort(im_fingers.flat)
    count = vals.shape[0]
    p01 = vals[int(round(0.01 * count))]
    p95 = vals[int(round(0.95 * count))]
    p99 = vals[int(round(0.999 * count))]

    # find highest peak
    counts, edges = np.histogram(vals, bins=50, density=True)
    i_max = np.argmax(counts)
    val_mode = (edges[i_max] + edges[i_max + 1]) / 2.0

    if p99 - p95 > 0.2:
        # thresh 1: assuming a symmetric distribution, mode+spread on low ed
        thresh1 = val_mode + (val_mode - p01) * 2

        # thresh 2: look for a distribution that drops very low
        below_02 = np.where(counts[i_max:] < 0.025)[0]
        if len(below_02) > 0:
            i = i_max + below_02[0]
            thresh2 = (edges[i] + edges[i + 1]) / 2.0
            thresh = min(thresh1, thresh2)
        else:
            thresh = thresh1
    else:
        thresh = im_fingers.max()

    thresh -= parameters.BRIGHT_AREA_MULTI_SENSITIVITY

    bright_areas = im_fingers - thresh
    bright_areas /= 0.5
    pixel_ops.ApplyThresholdLT_F32(bright_areas, bright_areas, 0.0, 0.0)
    pixel_ops.ApplyThresholdGT_F32(bright_areas, bright_areas, 1.0, 1.0)

    # create a full size image of bright lines
    bright_lines_full = np.zeros_like(im)
    bright_lines_full[row_nums, :] = bright_areas
    bright_lines_full = cv2.GaussianBlur(bright_lines_full, sigmaX=1, sigmaY=2, ksize=(0, 0))
    if 'ov_bright_area_u8' in features:
        features['ov_bright_area_u8'] = np.maximum((bright_lines_full * 255).astype(np.uint8),
                                                   features['ov_bright_area_u8'])
    else:
        features['ov_bright_area_u8'] = (bright_lines_full * 255).astype(np.uint8)
    features['bright_area_strength'] = bright_areas.mean() * 100
    features['bright_area_fraction'] = (pixel_ops.CountThresholdGT_F32(bright_areas, 0.1)) / float(
        im.shape[0] * im.shape[1])
    features['_bright_area_thresh'] = thresh

    if False:
        print features['bright_area_strength']
        print pixel_ops.CountThresholdGT_F32(bright_areas, 0.1) * 100 / float(im.shape[0] * im.shape[1])
        print features['bright_area_fraction']
        view = ImageViewer(im)
        ImageViewer(im_fingers)
        ImageViewer(bright_areas)
        plt.figure()
        plt.hist(im_fingers.flatten(), bins=50, normed=True)
        plt.plot((edges[:-1] + edges[1:]) / 2.0, counts)
        plt.vlines(thresh, 0, counts.max())
        view.show()


# @profile
def feature_extraction(im, features, already_cropped=False):
    t_start = timeit.default_timer()

    # rotation & cropping
    rotated = cropping.correct_cell_rotation(im, features, already_cropped=already_cropped)
    cropped = cropping.crop_cell(rotated, im, features, width=None, already_cropped=already_cropped)

    features['_cropped_f32'] = cropped
    features['im_cropped_u16'] = cropped.astype(np.uint16)
    h, w = cropped.shape

    if False:
        plt.figure()
        plt.plot(cropped.mean(axis=0))
        view = ImageViewer(im)
        ImageViewer(rotated)
        ImageViewer(cropped)
        view.show()

    # find fingers, busbars, etc
    cell.cell_structure(cropped, features)

    if False:
        view = ImageViewer(cropped)
        ImageViewer(features['bl_cropped_u8'])
        view.show()
        sys.exit()

    # normalise
    ip.histogram_percentiles(cropped, features, center_y=h // 2, center_x=w // 2, radius=features['wafer_radius'])
    cell.normalise(cropped, features)
    norm = features['im_norm']

    if False:
        view = ImageViewer(cropped)
        ImageViewer(norm)
        view.show()
        sys.exit()

    # full-size cell with no fingers/busbars
    cell.remove_cell_template(norm, features)

    if False:
        view = ImageViewer(norm)
        # ImageViewer(im_peaks)
        ImageViewer(features['im_no_fingers'])
        ImageViewer(features['im_no_figners_bbs'])
        view.show()
        sys.exit()

    if False:
        import os, glob
        folder = r"C:\Users\Neil\Desktop\crack_interp"
        fns = glob.glob(os.path.join(folder, "*.*"))
        fn_out = os.path.join(folder, "flat_%03d.png" % len(fns))

        im_out = (255 * ip.scale_image(features['im_no_figners_bbs'])).astype(np.uint8)
        ip.save_image(fn_out, im_out)

    if 'input_param_skip_features' not in features or int(features['input_param_skip_features']) != 1:
        # feature extraction
        finger_defects(features)
        bright_areas(features)
        efficiency_analysis(features)
        cell.multi_cracks(features)
        # firing_defects(features)

    # undo rotation
    if parameters.ORIGINAL_ORIENTATION and features['cell_rotated']:
        for feature in features.keys():
            if ((feature.startswith('im_') or feature.startswith('mask_') or
                     feature.startswith('map_') or feature.startswith('ov_') or
                     feature.startswith('bl_') or feature.startswith('mk_')) and features[feature].ndim == 2):
                features[feature] = features[feature].T[:, ::-1]
        if 'impure_edge_side' in features:
            if features['impure_edge_side'] == 0:
                features['impure_edge_side'] = 1
            elif features['impure_edge_side'] == 2:
                features['impure_edge_side'] = 3
            elif features['impure_edge_side'] == 1:
                features['impure_edge_side'] = 2
            elif features['impure_edge_side'] == 3:
                features['impure_edge_side'] = 0

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start


def finger_defects(features):
    # TODO: bright lines for top and bottom rows
    im = features['im_norm']
    row_nums = features['_finger_row_nums']

    ################
    # BRIGHT LINES #
    ################
    # highlight bright lines based on difference between lines above/below
    finger_im = im[row_nums, :].copy()

    # based on differences above & below
    rows_smooth = finger_im
    rows_filtered = np.zeros_like(rows_smooth)
    pixel_ops.FilterV(rows_smooth, rows_filtered)
    rows_filtered[:2, :] = finger_im[:2, :]
    rows_filtered[-2:, :] = finger_im[-2:, :]
    bright_lines = rows_smooth - rows_filtered

    bright_lines /= max(0.1, (0.5 - parameters.BRIGHT_LINES_MULTI_SENSITIVITY))

    # make less sensitive for multi
    bright_lines -= 0.25
    bright_lines /= 0.75

    bright_lines = ndimage.uniform_filter(bright_lines, size=(1, 5))
    pixel_ops.ClipImage(bright_lines, 0.0, 1.0)

    if False:
        plt.figure()
        plt.plot(finger_im[42, :])
        view = ImageViewer(finger_im)
        ImageViewer(im)
        ImageViewer(rows_filtered)
        ImageViewer(bright_lines)
        view.show()
        sys.exit()

    # get strength of each line (sum)
    # - need to insert lines between rows for computing CCs
    bl_cc = np.zeros((bright_lines.shape[0] * 2, bright_lines.shape[1]), np.uint8)
    bl_cc[::2, :] = bright_lines > 0
    ccs, num_ccs = ip.connected_components(bl_cc)
    ccs = ccs[::2, :]
    sums = ndimage.sum(bright_lines, ccs, range(num_ccs + 1))
    sums /= (float(bright_lines.shape[1]) / 100.0)
    line_strengths = np.take(sums, ccs)
    bright_lines[line_strengths < parameters.BRIGHT_LINES_MULTI_THRESHOLD] = 0
    features['bright_lines_count'] = (sums > parameters.BRIGHT_LINES_MULTI_THRESHOLD).sum()

    if False:
        print features['bright_lines_count']
        view = ImageViewer(bl_cc)
        ImageViewer(ccs)
        ImageViewer(line_strengths)
        ImageViewer(bright_lines)
        view.show()

    # create a full size image of bright lines
    bright_lines_full = np.zeros_like(im)
    bright_lines_full[row_nums, :] = bright_lines
    bright_lines_full = cv2.GaussianBlur(bright_lines_full, sigmaX=2, ksize=(0, 0))
    features['ov_bright_lines_u8'] = (bright_lines_full * 255).astype(np.uint8)
    features['bright_lines_strength'] = bright_lines.mean()
    features['bright_lines_length'] = (bright_lines > 0.5).sum()

    if False:
        im[row_nums, :] = 0
        view = ImageViewer(im)
        ImageViewer(finger_im)
        ImageViewer(bright_lines > 0.5)
        ImageViewer(bright_lines_full)
        view.show()
        sys.exit()

    if True:
        # want a 1-1 correspondence with bright lines, so:
        # - threshold based on sum of line
        # - add break to end with highest gradient
        f = np.array([[-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]], dtype=np.float32)
        edges_abs = np.abs(ndimage.correlate(finger_im, weights=f, mode="nearest"))
        cc_labels = np.where(sums > parameters.BRIGHT_LINES_MULTI_THRESHOLD)[0]
        max_pos = ndimage.maximum_position(edges_abs, ccs, cc_labels)
        breaks = np.zeros_like(finger_im, np.uint8)

        if len(max_pos) > 0:
            ys, xs = zip(*max_pos)
            breaks[np.array(ys, np.int32), np.array(xs, np.int32)] = 1
    else:
        ####################
        # BROKEN FINGERS 1 #
        ####################
        # - find breaks by comparing with fingers above/below (using bright lines image)
        # - the advantage of this approach is that it can find breaks near edges
        #   and busbars
        # - since there is a step 2, doesn't need to be very sensitive -- just find
        #   breaks near borders, which should be strong
        # - note that there also needs to be a gradient at that spot, otherwise we get
        #   some phantom breaks
        w = 11
        g = 3
        s = g + (w // 2)
        maxs = ndimage.maximum_filter(bright_lines, size=(1, w))
        mins = ndimage.minimum_filter(bright_lines, size=(1, w))
        diffs = np.maximum(np.roll(mins, s, axis=1) - np.roll(maxs, -s, axis=1),
                           np.roll(mins, -s, axis=1) - np.roll(maxs, s, axis=1))
        diffs = ndimage.gaussian_filter1d(diffs, sigma=3, axis=1)
        pixel_ops.ApplyThresholdLT_F32(diffs, diffs, 0.0, 0.0)
        local_maxes = np.logical_and((diffs > np.roll(diffs, 1, axis=1)),
                                     (diffs > np.roll(diffs, -1, axis=1)))
        diffs[~local_maxes] = 0
        f = np.array([[-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]], dtype=np.float32)
        edges_abs = np.abs(ndimage.correlate(finger_im, weights=f, mode="nearest"))
        breaks1 = ((diffs > parameters.BROKEN_FINGER_MULTI_THRESHOLD1) &
                   (edges_abs > parameters.BROKEN_FINGER_MULTI_EDGE_THRESHOLD))
        breaks1[:, :features['cell_edge_left']] = False
        breaks1[:, features['cell_edge_right']:] = False

        if False:
            view = ImageViewer(bright_lines)
            ImageViewer(diffs)
            ImageViewer(edges_abs)
            ImageViewer(breaks1)
            # ImageViewer(np.roll(mins, shift=s, axis=1) - np.roll(maxs, shift=-s, axis=1))
            view.show()
            sys.exit()

    # add to defect mask
    breaks_full = np.zeros_like(im)
    breaks_full[features["_finger_row_nums"], :] = breaks
    breaks_full = ndimage.binary_dilation(breaks_full, structure=ndimage.generate_binary_structure(2, 1), iterations=3)
    features['mk_finger_break_u8'] = breaks_full.astype(np.uint8)
    features['finger_break_count'] = breaks.sum()

    if False:
        view = ImageViewer(im)
        ImageViewer(bright_lines)
        # ImageViewer(break_strength)
        ImageViewer(breaks)
        ImageViewer(create_overlay(features))
        view.show()
        sys.exit()


def main():
    pass


if __name__ == '__main__':
    main()
