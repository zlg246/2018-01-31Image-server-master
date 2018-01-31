import numpy as np
from image_processing import ImageViewer
import image_processing as ip
import cell_processing as cell
import timeit
import cropping
import pixel_ops
import cv2
from scipy import ndimage, stats, optimize
import matplotlib.pylab as plt
import features_cz_cell as mono_cell
from scipy.spatial import distance
import features_multi_wafer as multi_wafer
import math
import parameters


def create_overlay(features):
    normed = features['im_cropped_u8']

    orig = normed.astype(np.int32)

    if False:
        view = ImageViewer(normed)
        view.show()

    b = orig
    g = orig
    r = orig

    if features['_cell_type'] == 'mono':
        pass
    elif features['_cell_type'] == 'multi':
        foreground = features['ov_dislocations_u8']
        b = orig + foreground
        g = orig - foreground
        r = orig - foreground

        impure = features['ov_impure2_u8']
        b -= impure
        g -= impure
        r += impure
    else:
        assert False

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)
    rgb = np.empty((normed.shape[0], normed.shape[1], 3), np.uint8)
    rgb[:, :, 0] = r.astype(np.uint8)
    rgb[:, :, 1] = g.astype(np.uint8)
    rgb[:, :, 2] = b.astype(np.uint8)

    # cracks
    if "mk_cracks_u8" in features:
        rgb = ip.overlay_mask(rgb, features['mk_cracks_u8'], 'r')

    if features['_cell_type'] == 'mono':
        # dark spots
        rgb = ip.overlay_mask(rgb, features['mk_dark_spots_outline_u8'], 'b')
        # dark areas
        if 'ov_dark_areas_u8' in features:
            dark = features["ov_dark_areas_u8"]
            rgb[:, :, 0] += dark
            rgb[:, :, 1] -= dark
            rgb[:, :, 2] += dark

    return rgb


def rough_crop(im, features):
    h, w = im.shape

    # rough crop to get rid of most of the background (for speed)
    cols = ndimage.gaussian_filter1d(im.mean(axis=0), 5)
    cols -= cols.min()
    cols /= cols.max()
    foreground_cols = np.where(cols > 0.025)[0]
    rows = ndimage.gaussian_filter1d(im.mean(axis=1), 5)
    rows -= rows.min()
    rows /= rows.max()
    foreground_rows = np.where(rows > 0.025)[0]
    left, right = foreground_cols[0], foreground_cols[-1]
    left, right = max(0, left - 10), min(w, right + 10)
    top, bottom = foreground_rows[0], foreground_rows[-1]
    top, bottom = max(0, top - 10), min(h, bottom + 10)

    features['_rough_bounds'] = (top, bottom, left, right)

    cropped = np.ascontiguousarray(im[top:bottom, left:right])

    return cropped

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
        ImageViewer(background1)
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
        ImageViewer(foreground1)
        ImageViewer(foreground2)
        ImageViewer(foreground)
        #ImageViewer(background_full)
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

    '''
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
    '''

    # top/bottom edges
    rows = np.apply_along_axis(stats.scoreatpercentile, 1, im, per=75)
    rows[rows > 1.0] = 1.0
    imp_profile = ndimage.gaussian_filter1d(rows, sigma=3, mode="nearest")
    imp_profile_r = imp_profile[::-1]
    row_defect = foreground.mean(axis=1)
    imp_v, ew_v, e_top = FindImpure(imp_profile, imp_profile_r, row_defect, debug=False)

    features['impure_edge_width'] = ew_v
    if e_top:
        features['impure_edge_side'] = 0
    else:
        features['impure_edge_side'] = 2

    impure = np.ones_like(im)
    impure[:, :] = np.c_[imp_v]

    if False:
        print features['impure_edge_width'], features['impure_edge_side']
        view = ImageViewer(impure)
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


def feature_extraction(im, features, skip_crop=False):
    t_start = timeit.default_timer()

    # rotation & cropping
    rotated = cropping.correct_stripe_rotation(im, features, already_cropped=skip_crop)
    cropped = cropping.crop_stripe(im, rotated, features, already_cropped=skip_crop)
    h, w = cropped.shape

    if False:
        view = ImageViewer(im)
        ImageViewer(cropped)
        view.show()

    features['_cropped_f32'] = cropped
    features['im_cropped_u16'] = cropped.astype(np.uint16)
    ip.histogram_percentiles(cropped, features)
    im_norm = cropped / features['hist_percentile_99']
    features['im_norm'] = im_norm
    pixel_ops.ApplyThresholdGT_F32(im_norm, im_norm, 1.0, 1.0)
    features['im_cropped_u8'] = np.round(im_norm * 255).astype(np.uint8)

    # TODO: finger/background mask
    features['bl_cropped_u8'] = np.zeros_like(im_norm, np.uint8)

    if False:
        view = ImageViewer(im)
        ImageViewer(features['im_cropped_u16'])
        ImageViewer(im_norm)
        view.show()

    if 'input_param_skip_features' in features and int(features['input_param_skip_features']) == 1:
        return

    features['_fingers_grid'] = False
    features['_busbar_cols'] = np.array([], np.int32)
    features['busbar_width'] = 0
    features['cell_edge_left'] = 10
    features['cell_edge_right'] = im_norm.shape[1] - 10
    features['mask_busbar_edges'] = np.zeros_like(im_norm, dtype=np.uint8)
    features['mask_busbar_filled'] = np.zeros_like(im_norm, dtype=np.uint8)
    features['wafer_middle_y'] = h // 2
    features['wafer_middle_x'] = w // 2
    features['wafer_radius'] = h
    features['_bright_area_thresh'] = 1
    features['cell_edge_tb'] = 0  # assume cropped already.
    cell.find_fingers(im_norm, features)
    cell.remove_cell_template(im_norm, features)

    # TODO: add background to mask
    features['bl_cropped_u8'] = np.zeros_like(im_norm, np.uint8)
    features['bl_cropped_u8'][features['_finger_row_nums'], :] = 4
    features['bl_cropped_u8'][im_norm < parameters.STRIPE_BACKGROUND_THRESH] = 1
    if False:
        view = ImageViewer(im_norm)
        ImageViewer(features['bl_cropped_u8'])
        view.show()

    if features['_cell_type'] == "multi":
        efficiency_analysis(features)
        cell.multi_cracks(features)
        features['ov_dislocations_u8'][:, :10] = 0
        features['ov_dislocations_u8'][:, -10:] = 0
    elif features['_cell_type'] == "mono":
        # calculate distance from wafer middle
        r, theta = np.empty_like(im_norm, np.float32), np.empty_like(im_norm, np.float32)
        pixel_ops.CenterDistance(r, theta, features['wafer_middle_y'], features['wafer_middle_x'])
        features['im_center_dist_im'] = r
        features['im_center_theta_im'] = theta

        cell.mono_cracks(features)
        mono_cell.dark_areas(features)
        mono_cell.dark_spots(features)
    else:
        print "ERROR -- Unknown mode: %s" % features['_cell_type']

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start

    return

def main():
    pass


if __name__ == '__main__':
    main()
