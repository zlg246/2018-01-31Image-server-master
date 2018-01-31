import numpy as np
from image_processing import ImageViewer
import image_processing as ip
import cv2
from scipy import ndimage, stats, interpolate
import sys
import timeit
import cell_processing as cell
from skimage import draw
import pixel_ops
import parameters
import cropping
from scipy import optimize
import matplotlib.pylab as plt
import features_cz_wafer as cz_wafer
from scipy.special import expit

import warnings


def finger_defects(features):
    im = features['im_norm']
    h, w = im.shape
    row_nums = features['_finger_row_nums']
    finger_im = im[row_nums, :]  # .copy()
    mask = features['bl_cropped_u8'][row_nums, :]
    bb_locs = features['_busbar_cols']

    # make a little more robust by averaging 3 locations:
    # - finger, and a little above/below
    offset = 1  # int(features['finger_period'] / 2.0)
    off_up = im[row_nums - offset, :]
    off_down = im[row_nums + offset, :]
    finger_im = (finger_im + off_up + off_down) / 3.0
    features['_finger_im'] = finger_im

    if False:
        view = ImageViewer(im)
        ImageViewer(finger_im)
        # ImageViewer(finger_im2)
        view.show()

    # make monotonic (don't care about local mins)
    mono_lr = np.empty_like(finger_im)
    mono_rl = np.empty_like(finger_im)
    pixel_ops.MakeMonotonicBBs(finger_im, mono_lr, bb_locs)
    pixel_ops.MakeMonotonicBBs(np.ascontiguousarray(finger_im[:, ::-1]), mono_rl,
                               np.ascontiguousarray(w - bb_locs[::-1]))
    mono_rl = mono_rl[:, ::-1]
    mono = np.minimum(mono_lr, mono_rl)

    if False:
        view = ImageViewer(im)
        # ImageViewer(normed)
        # ImageViewer(mono_lr)
        # ImageViewer(mono_rl)
        ImageViewer(mono)
        view.show()

    ####################
    # BROKEN FINGERS 1 #
    ####################
    # Detect intensity changes along a finger
    # - works best away from edges, and can handle breaks aligned in a column
    # 1. at a break, the min on bright side should be greater than max on dark side
    # 2. sharp local gradient, not a region with a steady (but high) gradient

    finger_filled = mono.copy()

    # 1 - brighter areas
    s = 25
    offset = s // 2 + 3
    maxs = ndimage.maximum_filter(finger_filled, size=(1, s), mode="nearest")
    pixel_ops.ApplyThresholdLT_F32(maxs, maxs, 0.1, 0.1)
    mins = ndimage.minimum_filter(finger_filled, size=(1, s), mode="nearest")
    d1 = np.roll(mins, offset, axis=1) / np.roll(maxs, -offset, axis=1)
    d2 = np.roll(mins, -offset, axis=1) / np.roll(maxs, offset, axis=1)
    break_strength = np.maximum(d1, d2)

    # 2 - sharp local drop
    f = np.array([[-1, -1, -1, -1, -1, 0, 1, 1, 1, 1, 1]], dtype=np.float32)
    edges_abs = np.abs(ndimage.correlate(mono, weights=f, mode="nearest"))
    edges_local = edges_abs - np.maximum(np.roll(edges_abs, 6, axis=1), np.roll(edges_abs, -6, axis=1))
    break_strength += edges_local
    features['_edges_abs'] = edges_abs

    if False:
        print parameters.BROKEN_FINGER_THRESHOLD1
        print parameters.BROKEN_FINGER_EDGE_THRESHOLD
        view = ImageViewer(finger_filled)
        # ImageViewer(maxs)
        # ImageViewer(mins)
        # ImageViewer(d1)
        # ImageViewer(d2)
        # ImageViewer(edges_local)
        ImageViewer(break_strength)
        view.show()
        sys.exit()

    # find local maxes
    breaks1 = ((break_strength >= np.roll(break_strength, 1, axis=1)) &
               (break_strength >= np.roll(break_strength, -1, axis=1)) &
               (break_strength > parameters.BROKEN_FINGER_THRESHOLD1)).astype(np.uint8)

    # For this detector, ignore breaks near edges, corners & busbars
    mask_background = mask == 8  # edges
    mask2 = ndimage.binary_dilation(mask_background, np.ones((1, s + 4), np.uint8))
    mask3 = ndimage.binary_dilation(mask2, np.ones((3, 1), np.uint8))
    corners = mask2.sum(axis=0) > 2
    mask2[:, corners] = mask3[:, corners]
    bb_mask = features['mask_busbar_filled'][row_nums, :]
    bb_mask = ndimage.binary_dilation(bb_mask, np.ones((1, 11), np.uint8))
    mask2 = (mask2 | bb_mask)
    breaks1[mask2] = 0

    if False:
        view = ImageViewer(im)
        ImageViewer(break_strength)
        ImageViewer(breaks1)
        view.show()
        sys.exit()

    ####################
    # BROKEN FINGERS 2 #
    ####################
    # - find breaks by comparing with fingers above/below (using bright lines image)
    # - the advantage of this approach is that it can find breaks near edges
    #   and busbars
    # - doesn't need to be very sensitive -- just find breaks near borders, which should be strong
    # - note that there also needs to be a gradient at that spot, otherwise we get
    #   some phantom breaks
    rows_filtered = np.zeros_like(mono)
    pixel_ops.FilterV(mono, rows_filtered)
    rows_filtered[:2, :] = finger_im[:2, :]
    rows_filtered[-2:, :] = finger_im[-2:, :]
    bright_lines = mono - rows_filtered

    mask_background = ((mask == 1) | (mask == 8))
    bright_lines[mask_background] = False
    pixel_ops.ClipImage(bright_lines, 0.0, 1.0)
    pixel_ops.ApplyThresholdLT_F32(mono, bright_lines, 0.4, 0.0)

    if False:
        view = ImageViewer(mono)
        ImageViewer(mask)
        ImageViewer(bright_lines)
        view.show()
        sys.exit()

    # filter bright lines
    min_length = w // 20
    cc_sums = np.zeros_like(bright_lines)
    breaks2 = np.zeros_like(bright_lines, np.uint8)
    pixel_ops.BrightLineBreaks(bright_lines, cc_sums, breaks2, 0.04,
                               parameters.BROKEN_FINGER_THRESHOLD2, min_length)

    if False:
        for r in range(cc_sums.shape[0]):
            if cc_sums[r, :].sum() == 0:
                continue
            print r
            plt.figure()
            plt.plot(bright_lines[r, :])
            plt.plot((cc_sums[r, :] > 0) * bright_lines[r, :].max())
            plt.figure()
            plt.plot(mono[r, :])
            plt.plot((cc_sums[r, :] > 0) * mono[r, :].max())
            plt.show()

    if False:
        view = ImageViewer(im)
        ImageViewer(bright_lines)
        ImageViewer(cc_sums)
        ImageViewer(breaks2)
        view.show()

    ###########
    # COMBINE #
    ###########
    # if there are 2+ close together, only keep one with strongest gradient

    if False:
        # check break source
        # green == independent lines (breaks1)
        # red == relative lines (breaks2)
        breaks_full = np.zeros_like(im)
        breaks_full[features["_finger_row_nums"], :] = breaks1
        breaks_full = ndimage.binary_dilation(breaks_full, structure=ndimage.generate_binary_structure(2, 1),
                                              iterations=3)
        rgb = ip.overlay_mask(im, breaks_full, 'g')

        breaks_full = np.zeros_like(im)
        breaks_full[features["_finger_row_nums"], :] = breaks2
        breaks_full = ndimage.binary_dilation(breaks_full, structure=ndimage.generate_binary_structure(2, 1),
                                              iterations=3)
        rgb = ip.overlay_mask(rgb, breaks_full, 'r')

        view = ImageViewer(rgb)
        view.show()
        sys.exit()

    breaks = (breaks1 | breaks2).astype(np.uint8)
    break_count = ndimage.correlate(breaks, weights=np.ones((1, 15), np.uint8), mode='constant')  # .astype(np.uinut8)
    pixel_ops.CombineBreaks(breaks, break_count, edges_abs)

    # ignore breaks in cell edge or background
    cell_template = features['bl_cropped_u8'][features['_finger_row_nums']]
    breaks[((cell_template == 1) | (cell_template == 8))] = 0

    # create break mask
    breaks_full = np.zeros_like(im, np.uint8)
    breaks_full[features["_finger_row_nums"], :] = breaks
    features['mk_finger_break_u8'] = breaks_full

    if False:
        print breaks_full.sum()
        view = ImageViewer(im)
        ImageViewer(finger_filled)
        ImageViewer(break_count)
        ImageViewer(breaks)
        view.show()
        sys.exit()


def finger_defect_features(features):
    breaks_full = features['mk_finger_break_u8']
    breaks = breaks_full[features['_finger_row_nums'], :]
    features['finger_break_count'] = breaks_full.sum()
    breaks_full = ndimage.binary_dilation(breaks_full, structure=ndimage.generate_binary_structure(2, 1), iterations=3)
    features['mk_finger_break_u8'] = breaks_full.astype(np.uint8)
    finger_im = features['_finger_im']
    edges_abs = features['_edges_abs']
    bb_locs = features['_busbar_cols']
    im = features['im_norm']

    if 'input_param_verbose' in features and features['input_param_verbose']:
        # compute feature based on finger break

        # create an edge/busbar mask
        cell_template = features['bl_cropped_u8'][features['_finger_row_nums'], :]
        cell_foreground = (cell_template != 4).astype(np.uint8)
        pixel_ops.FindCellMiddle(cell_foreground)

        # - position, distance to middle points, distance to furthest edge, strength
        ys, xs = np.where(breaks == 1)

        for i in range(len(ys)):
            y, x = ys[i], xs[i]

            row = cell_foreground[y, :]
            features['break%02d_y' % (i + 1)] = features['_finger_row_nums'][y]
            features['break%02d_x' % (i + 1)] = x
            features['break%02d_strength' % (i + 1)] = edges_abs[y, x]
            features['break%02d_middle_dist' % (i + 1)] = np.abs(np.where(row == 2)[0] - x).min()

            # find busbar/edge on left & right
            # TODO: not sure about logic when at busbar. guess this is worst case?
            edges = np.where(row == 1)[0]
            left = edges[np.where(edges < x)[0][-1]]
            right = edges[np.where(edges > x)[0][0]]
            if x - left > right - x:
                if breaks[y, left:x - 1].sum() > 0:
                    dist = -1
                else:
                    dist = x - left
            else:
                if breaks[y, x + 1:right].sum() > 0:
                    dist = -1
                else:
                    dist = right - x
            features['break%02d_edge_dist' % (i + 1)] = dist

    ################
    # BRIGHT LINES #
    ################
    # - if at least one break:
    #   - find strongest break. 0=dark side, 1=bright side
    bright_lines = np.zeros_like(finger_im)
    bbs = np.r_[0, bb_locs, finger_im.shape[1]]
    num_bright_lines = pixel_ops.BrokenFingerBrightLines(finger_im, bbs, edges_abs, breaks, bright_lines)
    features['bright_lines_count'] = num_bright_lines
    bright_lines = ndimage.uniform_filter(bright_lines, size=(1, 5))
    pixel_ops.ClipImage(bright_lines, 0.0, 1.0)

    # create a full size image of bright lines
    bright_lines_full = np.zeros_like(im)
    bright_lines_full[features['_finger_row_nums'], :] = bright_lines
    bright_lines_full = cv2.GaussianBlur(bright_lines_full, sigmaX=1, sigmaY=2, ksize=(0, 0))

    features['ov_bright_lines_u8'] = (bright_lines_full * 255).astype(np.uint8)
    features['bright_lines_strength'] = bright_lines.mean()
    features['bright_lines_length'] = (bright_lines > 0.5).sum()

    if False:
        print features['bright_lines_length']
        view = ImageViewer(im)
        # ImageViewer(normed)
        ImageViewer(bright_lines)
        ImageViewer(bright_lines > 0.5)
        # ImageViewer(breaks)
        ImageViewer(create_overlay(features))
        view.show()
        sys.exit()


# @profile
def dark_areas(features):
    # dark areas (not dark spots)
    im = features['im_norm']
    h, w = im.shape
    row_nums = features['_peak_row_nums']
    cell_mask = features['bl_cropped_u8'][row_nums, :]
    bb_locs = features['_busbar_cols']

    # fill edges & corners
    foreground = im[row_nums].copy()
    edge = features['cell_edge_tb']
    rr, cc = draw.circle_perimeter(features['wafer_middle_y'], features['wafer_middle_x'],
                                   int(round(features['wafer_radius'])) - edge)
    mask = (cc >= 0) & (cc < w) & np.in1d(rr, row_nums)
    lut = np.zeros(h, np.int32)
    lut[row_nums] = np.arange(len(row_nums))
    rr = rr[mask]
    cc = cc[mask]
    rr = np.take(lut, rr)
    pixel_ops.FillCorners(foreground, rr.astype(np.int32), cc.astype(np.int32))
    foreground[:, :edge] = np.c_[foreground[:, edge]]
    foreground[:, -edge:] = np.c_[foreground[:, -edge]]

    # create background
    # 1. make monotonic between edges & BBs
    mono_lr = np.empty_like(foreground)
    mono_rl = np.empty_like(foreground)
    pixel_ops.MakeMonotonicBBs(foreground, mono_lr, bb_locs)
    pixel_ops.MakeMonotonicBBs(np.ascontiguousarray(foreground[:, ::-1]), mono_rl,
                               np.ascontiguousarray(w - bb_locs[::-1]))
    mono_rl = mono_rl[:, ::-1]
    mono = np.minimum(mono_lr, mono_rl)
    background = mono
    da1 = background - foreground

    # fill BBs and flatten
    background2 = background.copy()
    pixel_ops.InterpolateBBs(background2, features['_busbar_cols'], features['busbar_width'])
    cols = background2.mean(axis=0)
    background2 -= np.r_[cols]
    rows = background2.mean(axis=1)
    background2 -= np.c_[rows]

    da2 = -1 * background2
    pixel_ops.ApplyThresholdLT_F32(da2, da2, 0.0, 0.0)

    if False:
        view = ImageViewer(foreground)
        ImageViewer(background)
        ImageViewer(background2)
        ImageViewer(da1)
        ImageViewer(da2)
        ImageViewer(da1 + da2)
        # plt.figure()
        # for x in range(0, background.shape[1], 50):
        #    plt.plot(background[:, x], label="Col %d"%(x))
        # plt.legend()
        view.show()
        sys.exit()

    dark_areas = da1 + da2
    dark_areas -= (0.1 - parameters.DARK_AREA_SENSITIVITY)
    dark_areas[(cell_mask == 1) | (cell_mask == 8)] = 0  # corners
    pixel_ops.ClipImage(dark_areas, 0.0, 1.0)
    dark_areas_full = np.empty_like(im)
    pixel_ops.ExpandFingers(dark_areas_full, dark_areas, features['_peak_row_nums'])

    if False:
        pixel_ops.ApplyThresholdGT_F32(features['im_center_dist_im'], dark_areas_full,
                                       features['wafer_radius'], 0)
        print features['wafer_radius']
        view = ImageViewer(dark_areas)
        ImageViewer(dark_areas_full)
        ImageViewer(features['im_center_dist_im'])
        view.show()

    dark_areas_full = cv2.GaussianBlur(dark_areas_full, ksize=(0, 0), sigmaX=1)

    # metrics
    mask_dark = (dark_areas_full > 0.2).astype(np.uint8)
    features['dark_area_strength'] = dark_areas_full.mean() * 100
    im_pl = features['_cropped_f32']
    darkPL = pixel_ops.MaskMean_F32(im_pl, mask_dark, 1)
    brightPL = pixel_ops.MaskMean_F32(im_pl, mask_dark, 0)
    features['dark_area_mean_PL'] = darkPL
    features['dark_area_fraction'] = mask_dark.mean()
    features['dark_area_PL_intensity_ratio'] = brightPL / max(1, darkPL)

    features['ov_dark_areas_u8'] = (dark_areas_full * 255).astype(np.uint8)

    if False:
        print features['dark_area_fraction'], features['dark_area_strength']
        # plt.figure()
        # plt.plot(cols)
        # plt.plot(cols_mono)
        view = ImageViewer(im)
        ImageViewer(foreground)
        ImageViewer(dark_areas)
        ImageViewer(mask_dark)
        ImageViewer(dark_areas_full)
        view.show()


# @profile
def firing_defects(features):
    im = features['im_norm']
    finger_rows = features['_finger_row_nums']
    # im_fingers = features['_finger_im']
    im_smooth = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=1)
    im_fingers = im_smooth[finger_rows, :]

    if False:
        view = ImageViewer(im)
        # view = ImageViewer(im_smooth)
        ImageViewer(im_fingers)
        view.show()
        sys.exit()

    # find depth of local minimums
    S = 5
    dips = np.minimum(np.roll(im_fingers, S, axis=1), np.roll(im_fingers, -S, axis=1)) - im_fingers
    pixel_ops.ApplyThresholdLT_F32(dips, dips, 0, 0)
    dips[:, features['mask_busbar_filled'][0, :]] = 0
    dips[:, :features['cell_edge_left']] = 0
    dips[:, features['cell_edge_right']:] = 0

    # TODO: add upper bound as well
    locs = ((im_fingers < np.roll(im_fingers, 1, axis=1)) &
            (im_fingers < np.roll(im_fingers, -1, axis=1)) &
            (dips > 0.02) & (dips < 0.05)).astype(np.float32)

    # count num dips in a region. ignore areas with only 1
    w = im.shape[1] // 10
    weights = np.ones(w, np.int32)
    local_count = ndimage.convolve1d(locs, weights, axis=1, mode="constant")
    dips_filtered = dips.copy()
    dips_filtered[local_count <= 1] = 0

    if False:
        view = ImageViewer(im)
        ImageViewer(dips)
        # ImageViewer(locs)
        # ImageViewer(local_count)
        # ImageViewer(dips_filtered)
        view.show()

    im_dots = cv2.GaussianBlur(dips_filtered, ksize=(0, 0), sigmaX=10, sigmaY=2)
    # im_dots -= 0.001

    if False:
        view = ImageViewer(dips_filtered)
        view = ImageViewer(im_dots)
        view.show()

    splotch = np.empty_like(im)
    pixel_ops.ExpandFingers(splotch, im_dots, finger_rows)
    splotch = ip.fast_smooth(splotch, sigma=10)
    features["_firing"] = splotch.copy()
    splotch *= 100.0 * (1.0 + parameters.FIRING_SENSITIVITY)
    pixel_ops.ClipImage(splotch, 0.0, 1.0)
    pixel_ops.ApplyThresholdGT_F32(features['im_center_dist_im'],
                                   splotch, features['wafer_radius'], 0)
    features["ov_splotches_u8"] = (splotch * 255).astype(np.uint8)

    # metrics
    if splotch.max() >= 0.2:
        features['firing_area_strength'] = splotch.mean() * 100
        mask_firing = (splotch > 0.2).astype(np.uint8)
        im_pl = features['_cropped_f32']
        firingPL = pixel_ops.MaskMean_F32(im_pl, mask_firing, 1)
        goodPL = pixel_ops.MaskMean_F32(im_pl, mask_firing, 0)
        features['firing_area_mean_PL'] = firingPL
        features['firing_area_fraction'] = mask_firing.mean()
        features['firing_area_PL_intensity_ratio'] = firingPL / max(1, goodPL)
    else:
        features['firing_area_strength'] = 0.0
        features['firing_area_mean_PL'] = 0.0
        features['firing_area_fraction'] = 0.0
        features['firing_area_PL_intensity_ratio'] = 0.0

    if False:
        print features['firing_area_strength']
        view = ImageViewer(im)
        ImageViewer(splotch, vmin=0, vmax=1)
        ImageViewer(mask_firing)
        f2 = {'im_cropped_u8': features['im_cropped_u8'],
              'ov_splotches_u8': features['ov_splotches_u8']}
        ImageViewer(create_overlay(f2))
        view.show()
        sys.exit()


def emitter_gridline_r(norm, features):
    bb_locs = features['_busbar_cols']
    h, w = norm.shape
    num_busbars = len(bb_locs)
    if num_busbars not in [2, 3]:
        print "WARNING: Can not compute emitter and gridline resistence"
        return

    if False:
        print features['_busbar_cols']
        view = ImageViewer(norm)
        view.show()
        sys.exit()

    ###################
    # Between busbars #
    ###################
    # Finger resistance

    def QuadraticFit(params, xs, curve):
        A, B, C = params
        ys = -1 * A * (xs - B) ** 2 + C
        return np.abs(curve - ys).mean()

    if num_busbars == 3:
        # global profile: average middles left and middle right
        # TODO: align profiles
        bb_left = norm[h // 4:-h // 4, bb_locs[0]:bb_locs[1]]
        bb_right = norm[h // 4:-h // 4, bb_locs[1]:bb_locs[2]]
        bb_L = np.median(bb_left, axis=0)
        bb_R = np.median(bb_right, axis=0)
        w = min(len(bb_L), len(bb_R))
        bbb = (bb_L[:w] + bb_R[:w]) / 2.0
    elif num_busbars == 2:
        bbb = np.median(norm[h // 4:-h // 4, bb_locs[0]:bb_locs[1]], axis=0)

    bbb = ndimage.gaussian_filter1d(bbb, sigma=2, mode="nearest")

    # fit parabola
    xs = np.arange(len(bbb))
    peak_pos = np.argmax(bbb)
    peak_val = bbb[peak_pos]
    params_op = optimize.fmin_powell(QuadraticFit, [0.0, peak_pos, peak_val],
                                     disp=0, args=(xs, bbb))
    fingerR, B, C = params_op
    ys = -1 * fingerR * (xs - B) ** 2 + C

    # compute goodness of fit
    p15 = int(round(bbb.shape[0] * 0.15))
    gof = np.abs(ys[p15:-p15] - bbb[p15:-p15]).mean()
    features['resistance_finger'] = fingerR
    features['resistance_finger_error'] = gof

    if False:
        print fingerR, gof

        plt.figure()
        plt.plot(bbb, label="x-profile")
        plt.plot(xs, ys, label="Fitted parabola")
        plt.legend()
        plt.show()
        sys.exit()

    ###################
    # Between fingers #
    ###################
    # aka: emitter or sheet resistance
    if num_busbars == 3:
        bb1, bb2, bb3 = bb_locs
        x1 = (bb1 + bb2) // 2
        x2 = (bb2 + bb3) // 2
        profile_L = norm[:, x1 - 5:x1 + 6].mean(axis=1)
        profile_R = norm[:, x2 - 5:x2 + 6].mean(axis=1)
    elif num_busbars == 2:
        # TODO: temp hack - handle this better
        bb1, bb2 = bb_locs
        x1 = (bb1 + bb2) // 2
        profile_L = norm[:, x1 - 5:x1 + 6].mean(axis=1)
        profile_R = profile_L

    if True:
        profile_avg = (profile_L + profile_R) / 2.0
        peaks = np.where((profile_avg > np.roll(profile_avg, 1)) &
                         (profile_avg > np.roll(profile_avg, -1)))[0][1:-1]
        peaks_sample = peaks[::5]
        # global measure
        # sample peaks (no need to use all of them)
        S = 100
        num_curves = len(peaks_sample) * 2
        curves = np.zeros((num_curves, S), np.float32)
        fp2 = (features['finger_period'] // 2) + 1

        # create a version with 10x resolutiobn
        f_super_L = interpolate.interp1d(np.arange(profile_L.shape[0]), profile_L,
                                         bounds_error=False, fill_value=0)
        f_super_R = interpolate.interp1d(np.arange(profile_R.shape[0]), profile_R,
                                         bounds_error=False, fill_value=0)

        # for each peak, interpolate a better peak
        for e, p in enumerate(peaks_sample):
            f_L = interpolate.interp1d([p - 1, p, p + 1], -1 * profile_L[p - 1:p + 2], kind="quadratic",
                                       bounds_error=False, fill_value=1.0)
            p_L = optimize.fmin_powell(f_L, p, disp=False)
            curves[e, :] = f_super_L(np.linspace(p_L - fp2, p_L + fp2, num=S))

            f_R = interpolate.interp1d([p - 1, p, p + 1], -1 * profile_R[p - 1:p + 2], kind="quadratic",
                                       bounds_error=False, fill_value=1.0)
            p_R = optimize.fmin_powell(f_R, p, disp=False)
            curves[e + peaks_sample.shape[0], :] = f_super_R(np.linspace(p_R - fp2, p_R + fp2, num=S))

        # find median of all peak, and cut off tails on each end
        peak_median = np.median(curves, axis=0)
        xs = np.linspace(0, fp2 * 2, S)
        s1 = np.argmin(peak_median[:S // 2])
        s2 = S // 2 + np.argmin(peak_median[S // 2:])
        peak_median = peak_median[s1:s2]
        xs = xs[s1:s2]

        # fit parabola
        peak_pos = xs[np.argmax(peak_median)]
        peak_val = peak_median[peak_pos]
        params_op = optimize.fmin_powell(QuadraticFit, [0.0, peak_pos, peak_val],
                                         disp=0, args=(xs, peak_median))
        emitterR, B, C = params_op
        ys = -1 * emitterR * (xs - B) ** 2 + C

        gof = np.abs(ys - peak_median).mean()
        features['resistance_emitter'] = emitterR
        features['resistance_emitter_error'] = gof

        if False:
            plt.figure()
            for i in range(num_curves):
                plt.plot(curves[i, :])
            plt.figure()
            plt.plot(xs, peak_median, label="y-profile")
            plt.plot(xs, ys, label="Fitted parabola")
            plt.legend()
            plt.show()
    else:
        finger_rows = features['_finger_row_nums'][1:-1]

        # local measure
        CREATE_MAP = True
        if CREATE_MAP:
            map_emitter = np.zeros_like(norm)
            map_gof = np.zeros_like(norm)

        for i in range(len(finger_rows) - 1):
            top = finger_rows[i] + 1
            bottom = finger_rows[i + 1]  # +1
            xs = np.arange(bottom - top)

            # left
            parabola = profile_L[top:bottom]
            coeff = np.polyfit(xs, parabola, 2)
            leftR = -1 * coeff[0]
            ys = np.poly1d(coeff)(xs)
            leftGOF = np.abs(ys - parabola).mean()

            if top == 791 and False:
                print leftR

                plt.figure()
                plt.plot(parabola, label="y-profile")
                plt.plot(xs, ys, label="Fitted parabola")
                plt.legend()
                plt.show()

            # right
            parabola = profile_R[top:bottom]
            coeff = np.polyfit(xs, parabola, 2)
            rightR = -1 * coeff[0]
            ys = np.poly1d(coeff)(xs)
            rightGOF = np.abs(ys - parabola).mean()

            if top == 147 and False:
                print rightR
                plt.figure()
                plt.plot(parabola, label="y-profile")
                plt.plot(xs, ys, label="Fitted parabola")
                plt.legend()
                plt.show()

            if CREATE_MAP:
                map_emitter[top - 1:bottom, bb1:bb2] = leftR
                map_emitter[top - 1:bottom, bb2:bb3] = rightR
                map_gof[top - 1:bottom, bb1:bb2] = leftGOF
                map_gof[top - 1:bottom, bb2:bb3] = rightGOF

        if False:
            view = ImageViewer(norm)
            ImageViewer(map_emitter)
            ImageViewer(map_gof)
            plt.show()

    return


def finger_shape(features):
    if 'DEBUG' in features:
        DEBUG = features['DEBUG']
    else:
        DEBUG = False

    # use an image that has been normalised to [0, 1]
    im = features['_cropped_f32'] / features['hist_percentile_99.9']

    if parameters.CELL_BB_MID_POINTS:
        locs = np.round((features['_busbar_cols'][:-1] + features['_busbar_cols'][1:]) / 2.0).astype(np.int32)
        pixel_ops.InterpolateBBs(im, locs, 3)

    im_finger = im[features['_peak_row_nums']]
    # firing = np.zeros_like(im_finger)

    if False:
        view = ImageViewer(im)
        ImageViewer(im_finger)
        plt.figure()
        plt.plot(im_finger.mean(axis=0))
        view.show()
        sys.exit()

    TRAINING_MODE = False
    if TRAINING_MODE:
        import os
        fn = "finger_shape.csv"
        # bb_locs = features['_busbar_cols']
        bb_locs = np.r_[0, features['_busbar_cols'], im.shape[1] - 1]
        with open(fn, 'a') as f:
            def on_click(event):
                tb = plt.get_current_fig_manager().toolbar
                if event.xdata is None: return
                if tb.mode != '':
                    print 'Not in click mode - turn of pan or zoom'
                    return

                if event.button == 1:
                    classification = "good"
                elif event.button == 3:
                    classification = "bad"
                else:
                    return

                y = round(event.ydata)
                x = int(round(event.xdata))
                i = np.searchsorted(bb_locs, x)

                left, right = bb_locs[i - 1], bb_locs[i]
                assert left < x < right
                vals = im_finger[y, left:right]

                if x < im_finger.shape[1] // 2:
                    vals = vals[::-1]

                plt.figure()
                plt.plot(vals)
                plt.show()

                valstr = ','.join(["%0.02f" % (v) for v in vals])

                f.write("%s,%s\n" % (classification, valstr))

            fig = plt.figure()
            fig.canvas.mpl_connect('button_press_event', on_click)
            plt.imshow(im_finger, cmap=plt.cm.gray, interpolation='nearest')
            plt.show()
        return

    if len(features['_busbar_cols']) > 1:
        bb_locs = features['_busbar_cols']
    else:
        # only 1 busbar, so add left and right edges
        bb_locs = np.r_[0, features['_busbar_cols'], im.shape[1] - 1]

    S = 15
    # analyse each finger independently
    peak_broken = []
    peak_fine = []
    finger_rs = []
    finger_maes = []

    if False:
        locs = []
        for bb in range(len(bb_locs) - 1):
            locs.append((bb_locs[bb] + bb_locs[bb + 1]) // 2)
        im_finger2 = im_finger.copy()
        pixel_ops.InterpolateBBs(im_finger2, np.array(locs, np.int32), 4)
        view = ImageViewer(im_finger)
        ImageViewer(im_finger2)
        view.show()
        im_finger = im_finger2

    for bb in range(len(bb_locs) - 1):
        segment = im_finger[:, bb_locs[bb] + S:bb_locs[bb + 1] - S]
        if segment.shape[1] == 0:
            continue
        xs = np.linspace(-1.0, 1.0, num=segment.shape[1])
        for y in range(5, segment.shape[0] - 5):
            # fit a quadratic curve
            bbb = segment[y, :]
            params = np.polyfit(xs, bbb, 2)

            f = np.poly1d(params)
            ys = f(xs)

            # calculate the mean absolute error between the actual pixel values and the fitted parabola
            mae = np.abs(ys - bbb).mean() / bbb.mean()
            sigmoid = expit((mae - 0.02) / 0.001)

            # save curevature & goodness of fit
            finger_rs.append(params[0] * -1)
            finger_maes.append(mae)

            if sigmoid > 0.7:
                peak_broken.append(bbb.max())
            elif sigmoid < 0.3:
                peak_fine.append(bbb.max())

            # if True and bb == 0 and y == 5:
            if False and sigmoid > 0.7:
                print mae, sigmoid
                print bb, y
                im_fin = im_finger.copy()
                im_fin[y - 2, bb_locs[bb] + S:bb_locs[bb + 1] - S] = 0
                im_fin[y + 2, bb_locs[bb] + S:bb_locs[bb + 1] - S] = 0
                ImageViewer(im_fin)
                plt.figure()
                plt.plot(xs, bbb, label="x-profile")
                plt.plot(xs, ys, label="Fitted parabola")
                plt.legend()
                plt.show()

    if len(finger_rs) > 0:
        features['resistance_finger'] = np.median(finger_rs)
        features['resistance_finger_error'] = np.median(finger_maes)
    else:
        features['resistance_finger'] = 0
        features['resistance_finger_error'] = 0

    if len(peak_broken) > 0 or len(peak_fine) > 0:
        range_min = np.array(peak_broken + peak_fine).min()
        range_max = np.array(peak_broken + peak_fine).max()
        range_vals = np.linspace(range_min, range_max, num=100)

        peak_broken = np.array(peak_broken)
        peak_fine = np.array(peak_fine)
        broken_percent = peak_broken.shape[0] * 100 / float(peak_broken.shape[0] + peak_fine.shape[0])
        features['fingers_non_para'] = broken_percent

        # compute distribution of peak vals of bad fingers
        bad_fingers_dist = None
        bad_mode = None
        if len(peak_broken) > 5:
            f_bad_fingers = stats.gaussian_kde(peak_broken)
            bad_fingers_dist = f_bad_fingers(range_vals)
            bad_maxs = np.where((bad_fingers_dist > np.roll(bad_fingers_dist, 1)) &
                                (bad_fingers_dist > np.roll(bad_fingers_dist, -1)) &
                                (bad_fingers_dist > 2.0))[0]
            if len(bad_maxs) > 0:
                bad_mode = range_vals[bad_maxs[np.argmax(bad_fingers_dist[bad_maxs])]]

        # compute distribution of peak vals of good fingers
        good_fingers_dist = None
        good_maxs = []
        good_mode, good_mode_i = None, None
        if len(peak_fine) > 5:
            f_good_fingers = stats.gaussian_kde(peak_fine)
            good_fingers_dist = f_good_fingers(range_vals)
            good_maxs = np.where((good_fingers_dist > np.roll(good_fingers_dist, 1)) &
                                 (good_fingers_dist > np.roll(good_fingers_dist, -1)) &
                                 (good_fingers_dist > 2))[0]

            if len(good_maxs) > 0:
                good_mode_i = good_maxs[np.argmax(good_fingers_dist[good_maxs])]
                good_mode = range_vals[good_mode_i]

        if False:
            ImageViewer(im_finger)
            plt.figure()
            if good_fingers_dist is not None:
                plt.plot(range_vals, good_fingers_dist, 'g')
            if bad_fingers_dist is not None:
                plt.plot(range_vals, bad_fingers_dist, 'b')
            plt.show()
            sys.exit()

        if broken_percent >= 70:
            if DEBUG:
                print "0"
            # lots of broken, use fixed threshold
            threshold = 0.7
        elif broken_percent >= 33 and (good_mode is None or bad_mode < good_mode):
            if DEBUG:
                print "1"
            # significant broken and "non-broken" fingers are brighter than broken ones
            threshold = 0.7
        elif len(good_maxs) >= 2:
            # 2+ peaks in good dist.
            # - perhaps some good fingers are being classified as bad
            if broken_percent < 30 and (bad_mode is None or good_mode < bad_mode):
                if DEBUG:
                    print "2"
                # mostly good and broken fingers brighter
                # - use the highest peak
                i = good_mode_i
                while True:
                    if (good_fingers_dist[i] < 0.5 or i == good_fingers_dist.shape[0] - 1 or
                            (good_fingers_dist[i] < good_fingers_dist[i + 1] and
                                     good_fingers_dist[i] < 3)): break
                    i += 1
                threshold = range_vals[i]
            elif broken_percent < 20 and bad_mode is not None and bad_mode < good_mode:
                if DEBUG:
                    print "2B"
                # mostly good but broken fingers darker
                # - use the highest peak
                threshold = (range_vals[good_maxs[-2]] + range_vals[good_maxs[-1]]) / 2.0
            elif bad_mode is not None:
                if DEBUG:
                    print "3"
                # quite a few bad, or broken fingers darker
                # - use first peak
                i = good_maxs[0]
                while True:
                    if (good_fingers_dist[i] < 0.5 or i == good_fingers_dist.shape[0] - 1 or
                            (good_fingers_dist[i] < good_fingers_dist[i + 1] and
                                     good_fingers_dist[i] < 3)): break
                    i += 1
                threshold = min(range_vals[i], bad_mode - 0.1)
            else:
                threshold = good_mode
        elif (broken_percent <= 33 >= 1 and (bad_mode is None or good_mode < bad_mode) or
                      broken_percent < 10) and len(good_maxs):
            # - majority good fingers, single peak & broken fingers (if they exist) are brighter
            # - base the treshold on the distribution of good finger peaks
            if DEBUG:
                print "4"

            # find main mode in good dist
            i = good_maxs[0]
            while True:
                if (good_fingers_dist[i] < 0.5 or
                            i == good_fingers_dist.shape[0] - 1 or
                        (good_fingers_dist[i] < good_fingers_dist[i + 1] and
                                 good_fingers_dist[i] < 3)):
                    break
                i += 1

            threshold = min(range_vals[good_maxs[-1]] + 0.2, range_vals[i])
        else:
            if broken_percent > 33:
                # roughly 50/50. broken fingers are brighter
                # - use middle between good & bad
                if DEBUG:
                    print "5"
                threshold = (good_mode + bad_mode) / 2.0
            else:
                if DEBUG:
                    print "6"
                # majority good. assume the "bad" ones aren't actually bad
                if good_mode_i is not None:
                    i = good_mode_i
                else:
                    i = np.argmax(good_fingers_dist)
                while True:
                    if (good_fingers_dist[i] < 0.5 or
                                i == good_fingers_dist.shape[0] - 1 or
                            (good_fingers_dist[i] < good_fingers_dist[i + 1] and
                                     good_fingers_dist[i] < 3)):
                        break
                    i += 1

                threshold = range_vals[i]
    else:
        threshold = 1.0

    threshold -= parameters.BRIGHT_AREA_SENSITIVITY

    if False:
        print "Percent broken: ", broken_percent
        print "Threshold:", threshold

        view = ImageViewer(im)
        plt.figure()
        m1, m2 = 0, 0
        if good_fingers_dist is not None:
            plt.plot(range_vals, good_fingers_dist, 'g', label="Not broken")
            m1 = good_fingers_dist.max()
        if bad_fingers_dist is not None:
            plt.plot(range_vals, bad_fingers_dist, 'r', label="Broken")
            m2 = bad_fingers_dist.max()
        plt.vlines(threshold, 0, max(m1, m2))
        plt.legend()
        view.show()

    # highlight areas brighter than threshold
    # features['bright_line_threshold'] = threshold
    if threshold < 1.0:
        bright_lines = (im[features['_peak_row_nums']] - threshold) / (1.0 - threshold)
    else:
        bright_lines = np.zeros_like(im_finger)
    bright_lines *= 0.5
    pixel_ops.ClipImage(bright_lines, 0, 1)

    # don't want to highlight single lines, so apply vertical median filter
    rows_filtered = np.zeros_like(bright_lines)
    pixel_ops.FilterV(bright_lines, rows_filtered)
    rows_filtered[:2, :] = bright_lines[:2, :]
    rows_filtered[-2:, :] = bright_lines[-2:, :]

    if False:
        view = ImageViewer(im_finger)
        ImageViewer(bright_lines)
        ImageViewer(rows_filtered)
        view.show()
        sys.exit()

    bright_lines = rows_filtered

    # create a full size image of bright areas
    bright_area_full = np.zeros_like(im)
    pixel_ops.ExpandFingers(bright_area_full, bright_lines, features['_peak_row_nums'])
    bright_area_full = cv2.GaussianBlur(bright_area_full, sigmaX=3, ksize=(0, 0))
    bright_u8 = (bright_area_full * 255).astype(np.uint8)
    if 'ov_bright_area_u8' in features:
        assert False
        # features['ov_bright_area_u8'] = np.maximum(bright_u8, features['ov_bright_area_u8'])
    else:
        features['ov_bright_area_u8'] = bright_u8
    features['bright_area_strength'] = bright_lines.mean() * 100

    # bright area metrics
    mask_bright = (bright_area_full > 0.1).astype(np.uint8)
    im_pl = features['_cropped_f32']
    brightPL = pixel_ops.MaskMean_F32(im_pl, mask_bright, 1)
    darkPL = pixel_ops.MaskMean_F32(im_pl, mask_bright, 0)
    features['bright_area_mean_PL'] = brightPL
    features['bright_area_fraction'] = mask_bright.mean()
    features['bright_area_PL_intensity_ratio'] = brightPL / max(1, darkPL)

    # firing problems
    # firing_full = np.zeros_like(im)
    # pixel_ops.ExpandFingers(firing_full, firing, features['_peak_row_nums'])


    if False:
        view = ImageViewer(im_finger)
        ImageViewer(bright_area_full)
        # view = ImageViewer(firing_full)
        ImageViewer(features['ov_bright_area_u8'])
        view.show()
        sys.exit()


# @profile
def wafer_features(features):
    gridless = features['im_no_figners_bbs']

    # full-size
    cz_wafer.process_rings(gridless, features, return_corrected=False)
    cz_wafer.rds(gridless, features)
    cz_wafer.radial_profile(gridless, features)
    cz_wafer.dark_middle(gridless, features)


def dark_spots(features):
    im = features['im_no_fingers']

    # shrink to standard size
    h, w = 300, 300
    im_small = cv2.resize(im, (h, w))

    dark_areas = np.zeros_like(im_small)
    pixel_ops.DarkSpots(im_small, dark_areas, 8)

    candidates = (dark_areas > parameters.DARK_SPOT_MIN_STRENGTH).astype(np.uint8)
    ip.remove_small_ccs(candidates, parameters.DARK_SPOT_MIN_SIZE)

    candidates = cv2.resize(candidates, (im.shape[1], im.shape[0]))
    candidates[features['mask_busbar_filled']] = 0

    dark_spots_outline = ndimage.binary_dilation(candidates, iterations=3).astype(np.uint8) - \
                         ndimage.binary_dilation(candidates, iterations=1).astype(np.uint8)
    features['mk_dark_spots_outline_u8'] = dark_spots_outline
    features['mk_dark_spots_filled_u8'] = candidates
    features['dark_spots_area_fraction'] = candidates.mean()
    dark_areas_no_noise = dark_areas - parameters.DARK_SPOT_MIN_STRENGTH
    pixel_ops.ApplyThresholdLT_F32(dark_areas_no_noise, dark_areas_no_noise, 0.0, 0.0)
    features['dark_spots_strength'] = dark_areas_no_noise.mean() * 10000
    features['dark_spots_count'] = ip.connected_components(candidates)[1]

    if False:
        print features['dark_spots_area_fraction']
        print features['dark_spots_strength']
        print features['dark_spots_count']
        rgb = ip.overlay_mask(im, dark_spots_outline)
        view = ImageViewer(rgb)
        ImageViewer(dark_areas)
        ImageViewer(dark_areas_no_noise)
        ImageViewer(candidates)
        view.show()


def feature_combination(features):
    if "ov_bright_area_u8" in features:
        # dilate to find defects near edges
        bright = (features['ov_bright_area_u8'] / 255.).astype(np.float32)
        kernel = np.ones((15, 15), np.uint8)
        bright = cv2.dilate(bright, kernel=kernel)
    else:
        bright = None
    if "ov_splotches_u8" in features:
        firing = (features['ov_splotches_u8'] / 255.).astype(np.float32)
    else:
        firing = None
    if "ov_dark_areas_u8" in features:
        dark = (features['ov_dark_areas_u8'] / 255.).astype(np.float32)
    else:
        dark = None

    # bright areas causing dark areas
    if bright is not None and dark is not None:
        weight = 5.0
        cols = bright.mean(axis=0)
        rows = bright.mean(axis=1)

        dark -= np.r_[cols] * weight
        dark -= np.c_[rows] * weight
        pixel_ops.ApplyThresholdLT_F32(dark, dark, 0.0, 0.0)

        if False:
            plt.figure()
            plt.plot(cols)
            plt.plot(rows)
            view = ImageViewer(bright)
            ImageViewer(features['ov_dark_areas_u8'] / 255.)
            ImageViewer(dark)
            view.show()

        features['ov_dark_areas_u8'] = (dark * 255).astype(np.uint8)

    # dark middles

    # remove dark spots & broken fingers in high firing areas
    if firing is not None:
        FIRING_THRESH = 0.1
        # if "mk_cracks_u8" in features:
        #    pixel_ops.ApplyThresholdGT_F32_U8(firing, features["mk_cracks_u8"], FIRING_THRESH, 0)
        if "mk_dark_spots_outline_u8" in features:
            pixel_ops.ApplyThresholdGT_F32_U8(firing, features["mk_dark_spots_outline_u8"], FIRING_THRESH, 0)

        if "mk_finger_break_u8" in features:
            # TODO: handle finger breaks differently: look at sum along finger, and remove all
            pixel_ops.ApplyThresholdGT_F32_U8(firing, features["mk_finger_break_u8"], FIRING_THRESH, 0)

        if False:
            view = ImageViewer(firing)
            view.show()

    # remove cracks, dark spots & broken fingers in bright areas
    if bright is not None:
        # expand a bit to get artifacts on corner

        BRIGHT_THRESH = 0.1
        # if "mk_cracks_u8" in features:
        #    pixel_ops.ApplyThresholdGT_F32_U8(bright, features["mk_cracks_u8"], BRIGHT_THRESH, 0)
        if "mk_finger_break_u8" in features:
            pixel_ops.ApplyThresholdGT_F32_U8(bright, features["mk_finger_break_u8"], BRIGHT_THRESH, 0)
        if "mk_dark_spots_outline_u8" in features:
            pixel_ops.ApplyThresholdGT_F32_U8(bright, features["mk_dark_spots_outline_u8"], BRIGHT_THRESH, 0)

        if False:
            view = ImageViewer(bright)
            view.show()

    # bright areas causing middle dark

    # overlapping cracks and dark spots
    if "mk_dark_spots_outline_u8" in features and "mk_cracks_u8" in features and \
                    features['mk_dark_spots_outline_u8'].max() > 0 and \
                    features['mk_cracks_u8'].max() > 0:
        ccs, num_ccs = ip.connected_components(features["mk_dark_spots_outline_u8"])
        remove_list = set(ccs[(features['mk_dark_spots_outline_u8'] == 1) & (features['mk_cracks_u8'] == 1)])
        lut = np.ones(num_ccs + 1, np.int32)
        lut[0] = 0
        lut[list(remove_list)] = 0
        removed = np.take(lut, ccs)

        if False:
            view = ImageViewer(features["mk_cracks_u8"])
            ImageViewer(ccs)
            ImageViewer(removed)
            view.show()

        features['mk_dark_spots_outline_u8'] = removed.astype(np.uint8)


# @profile
def feature_extraction(im, features, skip_crop=False):
    t_start = timeit.default_timer()

    # rotation & cropping
    rotated = cropping.correct_cell_rotation(im, features, already_cropped=skip_crop)
    cropped = cropping.crop_cell(rotated, im, features, width=None, already_cropped=skip_crop)

    features['_cropped_f32'] = cropped
    features['im_cropped_u16'] = cropped.astype(np.uint16)
    h, w = cropped.shape

    if False:
        view = ImageViewer(im)
        ImageViewer(rotated)
        ImageViewer(cropped)
        view.show()

    # find fingers, busbars, etc
    cell.cell_structure(cropped, features)

    if False:
        view = ImageViewer(im)
        ImageViewer(cropped)
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

    # remove mini busbars
    if parameters.CELL_BB_MID_POINTS:
        locs = np.round((features['_busbar_cols'][:-1] + features['_busbar_cols'][1:]) / 2.0).astype(np.int32)
        norm_no_min = norm.copy()
        pixel_ops.InterpolateBBs(norm_no_min, locs, 3)
        if False:
            print locs
            view = ImageViewer(norm)
            ImageViewer(norm_no_min)
            view.show()
            sys.exit()
        norm = norm_no_min
        features['im_norm'] = norm_no_min

    # full-size cell with no fingers/busbars
    cell.remove_cell_template(norm, features)

    if False:
        view = ImageViewer(norm)
        # ImageViewer(im_peaks)
        ImageViewer(features['im_no_fingers'])
        ImageViewer(features['im_no_figners_bbs'])
        view.show()
        sys.exit()

    if 'input_param_skip_features' not in features or int(features['input_param_skip_features']) != 1:
        # find cell features
        wafer_features(features)
        cell.mono_cracks(features)
        finger_defects(features)
        firing_defects(features)
        finger_shape(features)
        dark_areas(features)
        dark_spots(features)
        if 'input_param_no_post_processing' not in features or int(features['input_param_no_post_processing']) != 1:
            feature_combination(features)

        finger_defect_features(features)

    if False:
        # disable until we can do more tuning with Hunter and Juergen
        # - also should be merged with finger_shape
        emitter_gridline_r(norm, features)

    # undo rotation
    if parameters.ORIGINAL_ORIENTATION and features['cell_rotated']:
        for feature in features.keys():
            if ((feature.startswith('im_') or feature.startswith('mask_') or
                     feature.startswith('map_') or feature.startswith('ov_') or
                     feature.startswith('bl_') or feature.startswith('mk_')) and features[feature].ndim == 2):
                features[feature] = features[feature].T[:, ::-1]

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start


def create_overlay(features):
    im_u8 = features['im_cropped_u8']
    im_rgb = np.empty((im_u8.shape[0], im_u8.shape[1], 3), np.float32)
    im_rgb[:, :, :] = im_u8[:, :, np.newaxis]

    if True:
        # splotches
        if False and "ov_splotches_u8" in features:
            splotches = features["ov_splotches_u8"]
            im_rgb[:, :, 2] += splotches
            im_rgb[:, :, 1] -= 0.5 * splotches
            im_rgb[:, :, 0] -= 0.5 * splotches

        # bright lines/areas
        if "ov_bright_area_u8" in features:
            broken_fingers = features["ov_bright_area_u8"]  # *2
            im_rgb[:, :, 0] -= broken_fingers
            im_rgb[:, :, 1] += broken_fingers
            im_rgb[:, :, 2] -= broken_fingers

        if "ov_bright_lines_u8" in features:
            broken_fingers = features["ov_bright_lines_u8"]  # *2
            im_rgb[:, :, 0] -= broken_fingers
            im_rgb[:, :, 1] += broken_fingers
            im_rgb[:, :, 2] -= broken_fingers

    # cracks
    if "mk_cracks_u8" in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_cracks_u8'], 'r')

    if True:
        if 'mk_finger_break_u8' in features:
            im_rgb = ip.overlay_mask(im_rgb, features['mk_finger_break_u8'], 'b')

        if 'ov_dark_middle_u8' in features:
            impure = features["ov_dark_middle_u8"] // 2
            im_rgb[:, :, 0] += impure
            im_rgb[:, :, 1] += impure
            im_rgb[:, :, 2] -= impure

        if 'ov_dark_areas_u8' in features:
            impure = features["ov_dark_areas_u8"]
            im_rgb[:, :, 0] += impure
            im_rgb[:, :, 1] -= impure
            im_rgb[:, :, 2] += impure

        if "mk_dark_spots_outline_u8" in features:
            im_rgb = ip.overlay_mask(im_rgb, features['mk_dark_spots_outline_u8'], colour='r')

    im_rgb[im_rgb < 0] = 0
    im_rgb[im_rgb > 255] = 255
    im_rgb = im_rgb.astype(np.uint8)

    return im_rgb


def main():
    pass


if __name__ == "__main__":
    main()
