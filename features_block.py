import sys
import numpy as np
import image_processing as ip
from image_processing import ImageViewer
import cv2
import matplotlib.pylab as plt
import scipy.ndimage as ndimage
import scipy.stats as stats
import pixel_ops
from scipy import optimize
import math
import timeit
from scipy import interpolate
import parameters
import colormaps as cmaps


def block_crop(im, features):
    h, w = im.shape

    row_avg = ndimage.gaussian_filter1d(np.mean(im, axis=1), sigma=1).astype(np.float32)
    row_avg /= row_avg.max()
    row_profile = remove_dips(row_avg)
    row_profile -= row_profile.min()
    row_back = np.minimum(np.roll(row_profile, 5), np.roll(row_profile, -5))
    row_diff = np.abs(np.roll(row_profile, shift=3) - np.roll(row_profile, shift=-3))
    row_diff[:3] = 0
    row_diff[-3:] = 0
    row_diff[row_back > 0.1] = 0
    peaks = np.where((row_diff > np.roll(row_diff, 1)) &
                     (row_diff > np.roll(row_diff, -1)))[0]
    top = 0
    for p in peaks:
        if p > h // 2:
            break
        if row_diff[p] > row_diff[top] * 3.0:
            top = p
    bottom = h - 1
    for p in peaks[::-1]:
        if p < h // 2:
            break
        if row_diff[p] > row_diff[bottom] * 3.0:
            bottom = p

    col_avg = ndimage.gaussian_filter1d(np.mean(im[top:bottom], axis=0), sigma=1).astype(np.float32)
    col_avg /= col_avg.max()
    col_profile = remove_dips(col_avg)
    col_profile -= col_profile.min()
    col_back = np.minimum(np.roll(col_profile, 5), np.roll(col_profile, -5))
    col_diff = np.abs(np.roll(col_profile, shift=3) - np.roll(col_profile, shift=-3))
    col_diff[:3] = 0
    col_diff[-3:] = 0
    col_diff[col_back > 0.1] = 0
    peaks = np.where((col_diff > np.roll(col_diff, 1)) &
                     (col_diff > np.roll(col_diff, -1)))[0]
    col_peaks = peaks
    left = 0
    for p in peaks:
        if p > w // 2:
            break
        if col_diff[p] > col_diff[left] * 3.5:
            left = p
    right = w - 1
    for p in peaks[::-1]:
        if p < w // 2:
            break
        if col_diff[p] > col_diff[right] * 3.0:
            right = p

    if False:
        ImageViewer(im)
        plt.figure()
        plt.plot(col_profile)
        plt.plot(col_back)
        plt.plot(col_diff)
        plt.plot(col_peaks, col_diff[col_peaks], 'o')
        plt.vlines([left, right], 0, col_profile.max())
        plt.figure()
        plt.plot(row_profile)
        plt.plot(row_back)
        plt.plot(row_diff)
        plt.vlines([top, bottom], 0, row_profile.max())
        plt.show()

    features['_crop_bounds'] = (left, right, top, bottom)

    return im[top:bottom, left:right]


def create_overlay(features):
    normed = features['im_cropped_u8']
    background = features['ov_impure_u8']
    foreground = features['ov_defects_u8']

    orig = normed.astype(np.int32)

    if False:
        view = ImageViewer(orig)
        ImageViewer(background)
        ImageViewer(foreground)
        view.show()

    rgb = np.empty((background.shape[0], background.shape[1], 3), np.uint8)

    # foreground
    b = orig + foreground
    g = orig - foreground
    r = orig - foreground

    # background
    b -= background
    g -= background
    r += background

    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    rgb[:, :, 0] = r.astype(np.uint8)
    rgb[:, :, 1] = g.astype(np.uint8)
    rgb[:, :, 2] = b.astype(np.uint8)

    return rgb


# @profile
def block_foreground(im, features):
    per_col = features['_col_60']
    im_col = np.dot(np.ones((im.shape[0], 1), np.float32), per_col.reshape(1, per_col.shape[0]))

    per_row = features['_row_90']
    im_row = np.dot(per_row.reshape(im.shape[0], 1), np.ones((1, im.shape[1]), np.float32))

    background = ip.fast_smooth(np.minimum(im_col, im_row), sigma=5)
    foreground = background - im
    pixel_ops.ApplyThresholdLT_F32(foreground, foreground, 0, 0)
    pixel_ops.ApplyThresholdLT_F32(background, foreground, 0.3, 0)

    if False:
        view = ImageViewer(im, vmin=0, vmax=1)
        ImageViewer(im_col, vmin=0, vmax=1)
        ImageViewer(im_row, vmin=0, vmax=1)
        ImageViewer(background, vmin=0, vmax=1)
        ImageViewer(foreground, vmin=0, vmax=1)
        view.show()
        sys.exit()

    # skeletonized version of defects
    local_mins = np.zeros_like(foreground, np.uint8)
    f = cv2.GaussianBlur(foreground * -1, ksize=(0, 0), sigmaX=2)
    pixel_ops.LocalMins(f, local_mins)
    dis = ((local_mins == 1) & (foreground > 0.1)).astype(np.uint8)
    ys, xs = np.where(dis)
    pixel_ops.FastThin(dis, ys.copy(), xs.copy(), ip.thinning_lut)
    ip.remove_small_ccs(dis, 10)

    if False:
        crossing = np.zeros_like(dis)
        pixel_ops.ComputeCrossings(dis, crossing)

        junctions = crossing > 2
        struct = ndimage.generate_binary_structure(2, 2)
        junctions_d = ndimage.binary_dilation(junctions, struct)
        branches = dis.copy()
        branches[junctions_d] = 0

        # find branches that touch an end point
        ccs, num_ccs = ip.connected_components(branches)
        spurs = np.zeros_like(dis)
        for cc in set(ccs[crossing == 1]):
            if cc == 0: continue
            spurs[ccs == cc] = 1

        # sys.exit()
        remove = spurs.copy()
        ip.remove_small_ccs(remove, 10)
        removed = spurs - remove

        pruned = dis - removed
        crossing = np.zeros_like(dis)
        pruned = pruned.astype(np.uint8)
        pixel_ops.ComputeCrossings(pruned, crossing)
        pruned[crossing == 1] = 0

        dis = pruned

        rgb = ip.overlay_mask(im, dis, colour='b')
        ip.save_image("brick_lines_skeleton.png", dis)
        ip.save_image("brick_lines_overlay.png", rgb)
        view = ImageViewer(foreground, vmin=0, vmax=1)
        ImageViewer(rgb)
        view.show()
        sys.exit()

    # create a height-based profile of dislocation levels
    # - crop impure areas at top and bottom,
    imp = np.where(per_row < 0.5)[0]
    mid = len(per_row) // 2
    upper_half = imp[imp < mid]
    if len(upper_half) > 0:
        top = upper_half.max()
    else:
        top = 0
    lower_half = imp[imp > mid]
    if len(lower_half) > 0:
        bottom = lower_half.min()
    else:
        bottom = len(per_row)

    if False:
        plt.figure()
        plt.plot(per_row)
        plt.vlines([top, bottom], ymin=0, ymax=1)
        plt.show()
        sys.exit()
    foreground_pure = foreground[top:bottom, :]
    dislocation_profile = foreground_pure.mean(axis=0)
    dislocation_profile[per_col < 0.6] = 0
    dislocation_profile = ndimage.gaussian_filter1d(dislocation_profile, sigma=5)
    features['_dis_avg_height'] = dislocation_profile

    if False:
        view = ImageViewer(im, vmin=0, vmax=1)
        # ImageViewer(im_col, scale=0.5, vmin=0, vmax=1)
        # ImageViewer(im_row, scale=0.5, vmin=0, vmax=1)
        ImageViewer(background, vmin=0, vmax=1)
        ImageViewer(foreground, vmin=0, vmax=1)
        ImageViewer(foreground_pure, vmin=0, vmax=1)
        plt.figure()
        plt.plot(dislocation_profile)

        view.show()
        sys.exit()

    return foreground


def remove_dips(signal):
    l_to_r = np.zeros_like(signal)
    r_to_l = np.zeros_like(signal)
    pixel_ops.MakeMonotonic(signal, l_to_r)
    pixel_ops.MakeMonotonic(np.ascontiguousarray(signal[::-1]), r_to_l)
    r_to_l = r_to_l[::-1]
    no_dips = np.minimum(r_to_l, l_to_r)

    if False:
        plt.figure()
        plt.plot(signal)
        plt.plot(l_to_r)
        plt.plot(r_to_l)
        plt.plot(no_dips)
        plt.show()

    return no_dips


# @profile
def block_background(im, features):
    col_per = features['_col_90']

    if False:
        plt.figure()
        plt.plot(col_per)
        plt.show()
        sys.exit()

    # column profile
    cols_pl = ndimage.gaussian_filter1d(col_per, sigma=5)
    cols_pl *= features['norm_range']
    cols_pl += features['norm_lower']
    features['_pl_avg_height'] = cols_pl

    if False:
        # assume doping is monotonic, so skip dips
        cols_dipless = remove_dips(cols_pl)
        features['_pl_avg_height_dipless'] = cols_dipless

    cols = col_per.reshape((1, im.shape[1]))

    row_per = features['_row_90']
    rows = remove_dips(row_per)

    if False:
        plt.figure()
        plt.plot(row_per)
        plt.plot(rows)
        plt.show()
        sys.exit()
    rows = rows.reshape((im.shape[0], 1))

    background = np.dot(rows, cols)
    sigma = 20
    smooth = ip.fast_smooth(background, sigma=sigma)

    return smooth


def block_rotate(im_normed, features):
    im_small = cv2.GaussianBlur(im_normed, ksize=(0, 0), sigmaX=1)[::2, ::2]
    h, w = im_small.shape

    if False:
        view = ImageViewer(im_small)
        view.show()
        sys.exit()

    vals = []
    center_x = w // 2
    center_y = h // 2
    im_rotated = np.empty_like(im_small)
    rs = np.linspace(-1, 1, 11)
    for r in rs:
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), r, 1.0)
        im_rotated = cv2.warpAffine(im_small, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE, dst=im_rotated)

        rows = im_rotated.mean(axis=1)
        rows -= ndimage.gaussian_filter1d(rows, 1)
        cols = im_rotated.mean(axis=0)
        cols -= ndimage.gaussian_filter1d(cols, 1)

        vals.append(rows.std() + cols.std())

        if False:
            plt.figure()
            plt.plot(rows)
            plt.show()
            print rows.std()
            view = ImageViewer(im_rotated)
            view.show()
            # sys.exit()

    if False:
        print "Rotation: %0.02f" % rs[np.argmax(np.array(vals))]
        plt.figure()
        plt.plot(rs, vals)
        plt.show()

    rotation = rs[np.argmax(np.array(vals))]
    features['crop_rotation'] = rotation

    if abs(rotation) > 0.01:
        h, w = im_normed.shape
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        im_rotated = cv2.warpAffine(im_normed, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE, dst=im_rotated)
    else:
        im_rotated = im_normed

    if False:
        view = ImageViewer(im_normed)
        ImageViewer(im_rotated)
        view.show()

    return im_rotated


def robust_dislocations(smooth, impure, features):
    c = parameters.ROBUST_CROP
    smooth = np.ascontiguousarray(smooth[c:-c, c:-c])
    impure = np.ascontiguousarray(impure[c:-c, c:-c])

    struct = ndimage.generate_binary_structure(2, 1)

    # robust dislocation mask
    dog1 = (cv2.dilate(smooth, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                         (parameters.DOG_STRUCT_SIZE,
                                                          parameters.DOG_STRUCT_SIZE))) - smooth)
    dog2 = (ip.fast_smooth(smooth, sigma=parameters.DOG_SIGMA2) -
            cv2.GaussianBlur(smooth, (0, 0), parameters.DOG_SIGMA1, borderType=cv2.BORDER_REPLICATE))
    dog = dog1 + dog2

    IMP_THRESH = 0.4
    pixel_ops.ApplyThresholdLT_F32(impure, dog, IMP_THRESH, 0)

    if False:
        view = ImageViewer(dog1)
        ImageViewer(dog2)
        ImageViewer(dog1 + dog2)
        ImageViewer(dog)
        view.show()

    defect_mask = np.zeros_like(dog, np.uint8)
    DOG_THRESH = parameters.BLOCK_DISLOCATION_THRESH
    pixel_ops.ApplyThresholdGT_F32_U8(dog, defect_mask, DOG_THRESH, 1)
    num_pure_pixels = pixel_ops.CountThresholdGT_F32(impure, IMP_THRESH)
    defect_robust = (pixel_ops.CountEqual_U8(defect_mask, 1)) / float(num_pure_pixels)

    # compute surface area
    eroded = defect_mask - cv2.erode(defect_mask, struct.astype(np.uint8))
    defect_pixels = float(pixel_ops.CountEqual_U8(defect_mask, 1))
    if defect_pixels > 0:
        defect_surface = pixel_ops.CountEqual_U8(eroded, 1) / defect_pixels
    else:
        defect_surface = 0

    if False:
        print defect_robust, defect_surface
        view = ImageViewer(smooth)
        ImageViewer(defect_mask)
        ImageViewer(eroded)
        view.show()
        sys.exit()

    features['defect_robust_area_fraction'] = defect_robust
    features['defect_surface'] = defect_surface


def find_corners(im, features):
    h, w = im.shape
    bounds = features['_crop_bounds']
    (left, right, top, bottom) = bounds
    ps = []
    ps.append([top, left])  # top left
    ps.append([top, right])  # top right
    ps.append([bottom, right])  # bottom right
    ps.append([bottom, left])  # bottom left

    center_y, center_x = h // 2, w // 2

    theta = math.radians(features['crop_rotation'])
    shift_y, shift_x = 0, 0
    for p in ps:
        [y, x] = p
        # coordinates in image
        y_c = (y - center_y)
        x_c = (x - center_x)
        y_r = ((x_c * math.sin(theta)) + (y_c * math.cos(theta)))
        x_r = ((x_c * math.cos(theta)) - (y_c * math.sin(theta)))
        y_r += (center_y + shift_y)
        x_r += (center_x + shift_x)
        p[0] = int(round(y_r))
        p[1] = int(round(x_r))

    features['corner_tl_x'] = ps[0][1]
    features['corner_tl_y'] = ps[0][0]
    features['corner_tr_x'] = ps[1][1]
    features['corner_tr_y'] = ps[1][0]
    features['corner_br_x'] = ps[2][1]
    features['corner_br_y'] = ps[2][0]
    features['corner_bl_x'] = ps[3][1]
    features['corner_bl_y'] = ps[3][0]


# @profile
def feature_extraction(im, features, crop=True, skip_features=False):
    h, w = im.shape

    if im.dtype != np.float32:
        im = im.astype(np.float32)

    if crop:
        # cropping
        rotation_corrected = block_rotate(im, features)
        cropped_u16 = block_crop(rotation_corrected, features)
        bounds = features['_crop_bounds']
    else:
        rotation_corrected = im
        cropped_u16 = im
        bounds = (0, w - 1, 0, h - 1)
        features['_crop_bounds'] = bounds
        features['crop_rotation'] = 0

    # get original coordinates of the block corners
    find_corners(im, features)
    features['_rotation_corrected'] = rotation_corrected

    if False:
        view = ImageViewer(im)
        ImageViewer(cropped_u16)
        view.show()
        sys.exit()

    # normalisation
    vals = cropped_u16[::2, ::2].flat
    vals = np.sort(vals)
    min_val = vals[int(0.01 * vals.shape[0])]
    max_val = vals[int(0.99 * vals.shape[0])]
    features['norm_range'] = max_val - min_val
    features['norm_lower'] = min_val
    im_normed = (cropped_u16 - min_val) / (max_val - min_val)
    pixel_ops.ApplyThresholdLT_F32(im_normed, im_normed, 0.0, 0.0)

    cropped = im_normed
    croped_u8 = im_normed.copy()
    pixel_ops.ApplyThresholdGT_F32(croped_u8, croped_u8, 1.0, 1.0)
    features['im_cropped_u8'] = (croped_u8 * 255).astype(np.uint8)
    features['im_cropped_u16'] = cropped_u16.astype(np.uint16)

    if skip_features or ('input_param_skip_features' in features and int(features['input_param_skip_features']) == 1):
        return

    if False:
        view = ImageViewer(im)
        ImageViewer(cropped, vmin=0, vmax=1)
        view.show()
        sys.exit()

    # compute some row/column percentiles
    col_sorted = np.sort(cropped[::4, :], axis=0)
    features['_col_90'] = np.ascontiguousarray(col_sorted[int(round(0.9 * 0.25 * cropped.shape[0])), :])
    features['_col_60'] = np.ascontiguousarray(col_sorted[int(round(0.6 * 0.25 * cropped.shape[0])), :])
    row_sorted = np.sort(cropped[:, ::4], axis=1)
    features['_row_90'] = np.ascontiguousarray(row_sorted[:, int(round(0.9 * 0.25 * cropped.shape[1]))])

    # background
    background = block_background(cropped, features)

    # foreground
    foreground = block_foreground(cropped, features)

    # normalise background
    background /= background.max()

    # calculate metrics
    robust_dislocations(cropped, background, features)

    # dislocation area
    DIS_THRESH = 0.3
    dislocation_area = (pixel_ops.CountThresholdGT_F32(foreground, DIS_THRESH) /
                        float(foreground.shape[0] * foreground.shape[1]))
    impure_area = 1 - (pixel_ops.CountThresholdGT_F32(background, 0.5) /
                       float(foreground.shape[0] * foreground.shape[1]))

    # edge width
    l4 = background.shape[1] // 4
    profile = background[:, l4:-l4].mean(axis=1)
    fg = np.where(profile > parameters.BRICK_EDGE_THRESH)[0]
    if len(fg) > 0:
        left_width, right = fg[[0, -1]]
        right_width = len(profile) - right - 1
        edge_width = max(left_width, right_width)
        if edge_width < 0.05 * len(profile):
            edge_width = 0
    else:
        edge_width = 100
    features['edge_width'] = edge_width

    if False:
        print edge_width
        ImageViewer(cropped)
        plt.figure()
        plt.plot(profile)
        plt.show()

    if False:
        dislocations = np.zeros_like(foreground, dtype=np.uint8)
        pixel_ops.ApplyThresholdGT_F32_U8(foreground, dislocations, DIS_THRESH, 1)

        print features['defect_robust_area_fraction'], impure_area

        view = ImageViewer(im)
        ImageViewer(dislocations)
        ImageViewer(foreground)
        ImageViewer(background, vmin=0, vmax=1)
        view.show()
        # sys.exit()

    imp_cutoff = 0.55
    pixel_ops.ApplyThresholdGT_F32(background, background, imp_cutoff, imp_cutoff)
    background /= imp_cutoff
    background = np.log10(2 - background)

    dis_cutoff = 0.1
    foreground -= dis_cutoff
    foreground = np.clip(foreground, 0, 1)
    foreground *= 0.5

    features['ov_impure_u8'] = (background * 255).astype(np.uint8)
    features['ov_defects_u8'] = (foreground * 255).astype(np.uint8)
    features['_bounds'] = bounds
    pixel_ops.ClipImage(im_normed, 0, 1)
    features['dislocation_area_fraction'] = dislocation_area
    features['impure_area_fraction'] = impure_area

    return features


def fit_c_vals(im_pl, im_bulk, spline_plc):
    # trip top/bottom percentiles to make more robust to outliers
    im_pl_sort = np.sort(im_pl, axis=0)
    im_bulk_sort = np.sort(im_bulk, axis=0)
    p = max(5, int(round(0.01 * im_pl.shape[0])))
    im_pl = im_pl_sort[p:-p, :]
    im_bulk = im_bulk_sort[p:-p, :]

    bulk_cols = im_bulk.mean(axis=0)
    c_vals = []
    for col in range(im_pl.shape[1]):
        bulk_val = bulk_cols[col]
        pl_vals = im_pl[:, col]

        # optimize C
        def diff(params):
            c = params[0]
            return abs(bulk_val - (interpolate.splev(pl_vals * c, spline_plc).mean()))

        if len(c_vals) == 0:
            init_c = 1
        else:
            init_c = c_vals[-1]
        c_vals.append(optimize.fmin(diff, init_c, disp=0, xtol=1, ftol=0.005)[0])

    return c_vals


def find_cuts(cropped, mode, features):
    w = cropped.shape[1]
    bt_profile = np.median(cropped, axis=0)
    bt_profile = ndimage.gaussian_filter1d(bt_profile, 2)

    if mode == "SP":
        bt_profile /= bt_profile.max()
        bottom_thresh = parameters.CUTTING_THRESH_BOTTOM_SP
        top_thresh = parameters.CUTTING_THRESH_TOP_SP
        features['_sp_profile'] = bt_profile
    elif mode == "plir":
        bottom_thresh = parameters.CUTTING_THRESH_BOTTOM_PLIR
        top_thresh = parameters.CUTTING_THRESH_TOP_PLIR
        features['_plir_profile'] = bt_profile
    else:
        print "ERROR: unknown cutting mode"

    peak_pos = np.argmax(bt_profile)
    first_half = bt_profile[:peak_pos][::-1]
    locs = np.where(first_half < bottom_thresh)[0]
    if len(locs) == 0:
        bottom_cut = 0
    else:
        bottom_cut = len(first_half) - locs[0] - 1

    second_half = bt_profile[peak_pos:]
    locs = np.where(second_half < top_thresh)[0]
    if len(locs) == 0:
        top_cut = w - 1
    else:
        top_cut = peak_pos + locs[0]

    if features['marker_loc'] > 0:
        b = features['marker_loc'] - bottom_cut
        t = top_cut - features['marker_loc']
    else:
        b, t = w - bottom_cut - 1, w - top_cut - 1

    if False:
        print mode
        print features['marker_loc'], b, t
        plt.figure()
        plt.plot(bt_profile)
        plt.vlines([bottom_cut, top_cut], 0, bt_profile.max())
        plt.show()

    return b, t


def find_marker(im):
    h2 = im.shape[0] // 2
    h5 = im.shape[0] // 5
    im_mid = im[h2 - h5:h2 + h5]

    profile = np.median(im_mid, axis=0)
    profile /= profile.max()
    profile = ndimage.gaussian_filter1d(profile, 1)

    if im.shape[1] > 2000:
        s = 10
    else:
        s = 5

    # rough estimate
    dips = np.minimum(np.roll(profile, s), np.roll(profile, -s)) - profile
    dips[:10] = 0
    dips[-10:] = 0
    peak_loc = np.argmax(dips)
    dip_strength = dips[peak_loc]

    if dip_strength < 0.1:
        return 0

    if False:
        print dip_strength
        view = ImageViewer(im_mid)
        plt.figure()
        plt.plot(profile)
        plt.vlines([peak_loc], 0, profile.max())
        plt.figure()
        plt.plot(dips)
        plt.vlines([peak_loc], 0, profile.max())
        plt.show()
        # sys.exit()

    # fine tune
    win_h, win_w = 20, 5
    win = im[h2 - win_h:h2 + win_h + 1, peak_loc - win_w:peak_loc + win_w + 1]
    signal = np.median(win, axis=0)
    signal = ndimage.gaussian_filter1d(signal, 1)
    xs = np.arange(len(signal))
    f_cubic = interpolate.interp1d(xs, signal, kind='quadratic', bounds_error=False, fill_value=0)

    lowest_point = optimize.fmin(f_cubic, len(signal) // 2, disp=False)[0]

    marker_location = peak_loc - win_w + lowest_point

    if False:
        view = ImageViewer(im)
        ImageViewer(win)
        plt.figure()
        plt.plot(signal)
        xs = np.linspace(0, len(signal) - 1, 100)
        plt.plot(xs, f_cubic(xs))
        plt.plot(lowest_point, f_cubic(lowest_point), 'o')
        plt.figure()
        plt.plot(profile)
        plt.vlines([marker_location], 0, profile.max())
        plt.show()

    return int(round(marker_location))


def plir2(im_sp, im_lp, features, spline_plir, spline_sp):
    pixel_ops.ApplyThresholdLT_F32(im_sp, im_sp, 1.0, 1.0)
    pixel_ops.ApplyThresholdLT_F32(im_lp, im_lp, 1.0, 1.0)
    if im_sp.shape != im_lp.shape:
        print im_sp.shape, im_lp.shape
        assert False

    im_sp = im_sp.astype(np.float64)
    im_lp = im_lp.astype(np.float64)

    if False:
        view = ImageViewer(im_sp)
        ImageViewer(im_lp)
        view.show()
        sys.exit()

    # register short and long pass images
    def register(signal1, signal2, debug=False):
        bandpass1 = ndimage.gaussian_filter1d(signal1, sigma=10) - ndimage.gaussian_filter1d(signal1, sigma=3)
        bandpass2 = ndimage.gaussian_filter1d(signal2, sigma=10) - ndimage.gaussian_filter1d(signal2, sigma=3)
        offsets = range(-10, 11)
        fits = [(np.roll(bandpass1, shift=s) * bandpass2).mean() for s in offsets]
        optimal_shift = offsets[np.argmax(fits)]

        if debug:
            plt.figure()
            plt.plot(signal1)
            plt.plot(signal2)
            plt.figure()
            plt.plot(offsets, fits)
            plt.figure()
            plt.plot(np.roll(bandpass1, shift=optimal_shift))
            plt.plot(bandpass2)
            plt.show()

        return optimal_shift

    c = np.argmax(im_sp.mean(axis=0))
    profile_sp = im_sp[:, c - 10:c + 11].mean(axis=1)
    profile_lp = im_lp[:, c - 10:c + 11].mean(axis=1)
    shift_v = register(profile_sp, profile_lp, debug=False)
    if False:
        print c, shift_v
        view = ImageViewer(im_lp)
        ImageViewer(im_sp)
        ImageViewer(np.roll(im_sp, shift=shift_v, axis=0))
        view.show()
        sys.exit()
    im_sp = np.roll(im_sp, shift=shift_v, axis=0)

    # compute PL (ratio of LP to SP)
    plir = im_lp / im_sp

    if False:
        t = stats.scoreatpercentile(plir, per=99)
        print plir.min(), t, plir.max()
        plir[plir > t] = t
        view = ImageViewer(im_lp)
        ImageViewer(im_sp)
        ImageViewer(plir)
        view.show()
        sys.exit()

    # Get cropping and rotation parameters (based on short pass)
    # normalisation - find the 99th percentile
    vals = im_sp[::4, ::4].flat
    vals = np.sort(vals)
    min_val = vals[int(0.025 * vals.shape[0])]
    max_val = vals[int(0.975 * vals.shape[0])]
    features['norm_range'] = max_val - min_val
    features['norm_lower'] = min_val
    im_normed_temp = (im_sp - min_val) / (max_val - min_val)
    rotated_temp = block_rotate(im_normed_temp, features)

    # get crop bounds crop
    block_crop(rotated_temp, features)
    rotation = features['crop_rotation']
    x1, x2, y1, y2 = features['_crop_bounds']

    if 'input_param_skip_features' in features and int(features['input_param_skip_features']) == 1:
        return True

    if False:
        cropped_temp = rotated_temp[y1:y2, x1:x2]
        view = ImageViewer(im_sp)
        ImageViewer(rotated_temp)
        ImageViewer(cropped_temp)
        view.show()

    # correct and crop plir image (using params from short pass)
    if abs(rotation) > 0.01:
        h, w = plir.shape
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        plir_rotated = cv2.warpAffine(plir, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        sp_rotated = cv2.warpAffine(im_sp, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        lp_rotated = cv2.warpAffine(im_lp, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    else:
        plir_rotated = plir
        sp_rotated = im_sp
        lp_rotated = im_lp

    features['marker_loc'] = find_marker(sp_rotated[y1:y2, x1:x2])

    plir_cropped = plir_rotated[y1:y2, x1:x2]
    plir_cropped = np.ascontiguousarray(plir_cropped)

    # continue with plir
    _, upper = ip.get_percentile(plir_cropped, 0.005)
    pixel_ops.ClipImageF64(plir_cropped, 0, upper)

    if False:
        _, upper = ip.get_percentile(plir, 0.005)
        pixel_ops.ClipImageF64(plir, 0, upper)
        print plir_cropped.min(), plir_cropped.dtype
        view = ImageViewer(plir)
        ImageViewer(plir_cropped)
        view.show()
        sys.exit()

    tau_bulk_plir = interpolate.splev(plir_cropped.flatten(), spline_plir).reshape(plir_cropped.shape)
    _, upper = ip.get_percentile(tau_bulk_plir, 0.0001)
    pixel_ops.ClipImageF64(tau_bulk_plir, 0.1, upper)

    if False:
        ImageViewer(im_sp)
        ImageViewer(im_lp)
        plt.figure()
        plt.imshow(plir_cropped, cmap=cmaps.viridis)
        plt.colorbar()
        plt.figure()
        plt.imshow(tau_bulk_plir, cmap=cmaps.viridis)
        plt.colorbar()
        plt.show()
        sys.exit()

    # dislocation and impure processing for PL image
    cropped_sp = sp_rotated[y1:y2, x1:x2]
    cropped_lp = lp_rotated[y1:y2, x1:x2]

    # compute c-values
    c_vals = fit_c_vals(cropped_sp, tau_bulk_plir, spline_sp)
    doping = ndimage.gaussian_filter1d(c_vals, sigma=2, mode="reflect")

    if False:
        ImageViewer(cropped_sp)
        plt.figure()
        plt.plot(c_vals)
        plt.plot(doping)
        plt.show()

    features['_C_vals'] = doping.astype(np.float32)
    sp_dope = cropped_sp * np.r_[doping]
    sp_dope[sp_dope > spline_sp[0][-1]] = spline_sp[0][-1]
    tau_bulk_full = interpolate.splev(sp_dope.flatten(), spline_sp).astype(np.float32).reshape(
        cropped_sp.shape)

    # pixel_ops.ApplyThresholdLT_F32(tau_bulk_full, tau_bulk_full, 0.1, 0.1)
    _, upper_p = ip.get_percentile(tau_bulk_full, 0.0001)
    pixel_ops.ClipImage(tau_bulk_full, 0.1, upper_p)

    features['im_tau_bulk_f32'] = tau_bulk_full
    features['im_tau_bulk_u8'] = (ip.scale_image(tau_bulk_full) * 255).astype(np.uint8)
    features['im_cropped_sp_u8'] = (ip.scale_image(cropped_sp) * 255).astype(np.uint8)
    features['im_cropped_sp_u16'] = np.round(cropped_sp).astype(np.uint16)
    features['im_cropped_lp_u16'] = np.round(cropped_lp).astype(np.uint16)

    if False:
        if True:
            _, upper_p = ip.get_percentile(tau_bulk_full, 0.001)
            pixel_ops.ClipImage(tau_bulk_full, 0.1, upper_p)
        else:
            tau_bulk_full = np.log(tau_bulk_full)
        ImageViewer(cropped_sp)
        ImageViewer(sp_dope)
        ImageViewer(tau_bulk_plir)
        ImageViewer(tau_bulk_full)
        plt.figure()
        plt.plot(tau_bulk_plir.mean(axis=0))
        plt.plot(tau_bulk_full.mean(axis=0))
        if False:
            plt.figure()
            plt.plot(doping)
            plt.plot(c_vals)
        plt.show()

    return True


# register short and long pass images
def register(signal1, signal2, debug=False):
    bandpass1 = ndimage.gaussian_filter1d(signal1, sigma=10) - ndimage.gaussian_filter1d(signal1, sigma=3)
    bandpass2 = ndimage.gaussian_filter1d(signal2, sigma=10) - ndimage.gaussian_filter1d(signal2, sigma=3)
    offsets = range(-10, 11)
    fits = [(np.roll(bandpass1, shift=s) * bandpass2).mean() for s in offsets]
    optimal_shift = offsets[np.argmax(fits)]

    if debug:
        plt.figure()
        plt.plot(signal1)
        plt.plot(signal2)
        plt.figure()
        plt.plot(offsets, fits)
        plt.figure()
        plt.plot(np.roll(bandpass1, shift=optimal_shift))
        plt.plot(bandpass2)
        plt.show()

    return optimal_shift


# @profile
def plir(im_sp, im_lp, im_pl, features, spline_plir, spline_plc):
    t_start = timeit.default_timer()

    pixel_ops.ApplyThresholdLT_F32(im_sp, im_sp, 1.0, 1.0)
    pixel_ops.ApplyThresholdLT_F32(im_lp, im_lp, 1.0, 1.0)
    pixel_ops.ApplyThresholdLT_F32(im_pl, im_pl, 1.0, 1.0)
    if im_sp.shape != im_lp.shape:
        print im_sp.shape, im_lp.shape
        assert False
    im_sp = im_sp.astype(np.float64)
    im_lp = im_lp.astype(np.float64)
    im_pl = im_pl.astype(np.float64)

    if False:
        view = ImageViewer(im_sp)
        ImageViewer(im_lp)
        ImageViewer(im_pl)
        view.show()
        sys.exit()

    # vertical registration
    c = np.argmax(im_sp.mean(axis=0))
    profile_sp = im_sp[:, c - 10:c + 11].mean(axis=1)
    profile_lp = im_lp[:, c - 10:c + 11].mean(axis=1)
    shift_v = register(profile_sp, profile_lp, debug=False)
    if False:
        print c, shift_v
        view = ImageViewer(im_lp)
        ImageViewer(im_sp)
        ImageViewer(np.roll(im_sp, shift=shift_v, axis=0))
        view.show()
        sys.exit()
    im_sp = np.roll(im_sp, shift=shift_v, axis=0)

    # compute plir (ratio of LP to SP)
    plir = im_lp / im_sp

    if False:
        t = stats.scoreatpercentile(plir, per=90)
        print plir.min(), t, plir.max()
        plir[plir > t] = t
        view = ImageViewer(im_lp)
        ImageViewer(im_sp)
        ImageViewer(plir)
        view.show()
        sys.exit()

    # Get cropping and rotation parameters (based on short pass)
    vals = im_sp[::2, ::2].flat
    vals = np.sort(vals)
    min_val = vals[int(0.025 * vals.shape[0])]
    max_val = vals[int(0.975 * vals.shape[0])]
    features['norm_range'] = max_val - min_val
    features['norm_lower'] = min_val
    im_normed_temp = (im_sp - min_val) / (max_val - min_val)
    rotated_temp = block_rotate(im_normed_temp, features)
    block_crop(rotated_temp, features)
    rotation = features['crop_rotation']
    x1, x2, y1, y2 = features['_crop_bounds']

    if False:
        cropped_temp = rotated_temp[y1:y2, x1:x2]
        view = ImageViewer(im_sp)
        ImageViewer(rotated_temp)
        ImageViewer(cropped_temp)
        view.show()

    if 'input_param_skip_features' in features and int(features['input_param_skip_features']) == 1:
        return True

    # rotate all images
    if abs(rotation) > 0.01:
        h, w = plir.shape
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        plir_rotated = cv2.warpAffine(plir, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REPLICATE)
        sp_rotated = cv2.warpAffine(im_sp, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        lp_rotated = cv2.warpAffine(im_lp, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
        h, w = im_pl.shape
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        nf_rotated = cv2.warpAffine(im_pl, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    else:
        plir_rotated = plir
        sp_rotated = im_sp
        lp_rotated = im_lp
        nf_rotated = im_pl

    # find marker location
    features['marker_loc'] = find_marker(sp_rotated[y1:y2, x1:x2])
    nf_to_sp_ratio = nf_rotated.shape[1] / float(sp_rotated.shape[1])
    features['marker_loc'] *= nf_to_sp_ratio

    # crop plir image
    cropped_plir = plir_rotated[y1:y2, x1:x2]
    cropped_plir = np.ascontiguousarray(cropped_plir)
    cropped_sp = sp_rotated[y1:y2, x1:x2]
    cropped_lp = lp_rotated[y1:y2, x1:x2]

    if False:
        _, upper = ip.get_percentile(cropped_plir, 0.005)
        pixel_ops.ClipImageF64(cropped_plir, 0, upper)
        print cropped_plir.min(), cropped_plir.dtype
        view = ImageViewer(plir)
        ImageViewer(cropped_plir)
        ImageViewer(cropped_sp)
        view.show()
        sys.exit()

    # convert plir image to bulk image
    tau_bulk_plir = interpolate.splev(cropped_plir.flatten(), spline_plir).reshape(cropped_plir.shape)
    _, upper = ip.get_percentile(tau_bulk_plir, 0.0001)
    pixel_ops.ClipImageF64(tau_bulk_plir, 0.1, upper)

    if False:
        ImageViewer(im_sp)
        ImageViewer(im_lp)
        plt.figure()
        plt.imshow(cropped_plir)
        plt.colorbar()
        plt.figure()
        plt.imshow(tau_bulk_plir)
        plt.colorbar()
        plt.show()
        sys.exit()

    # zoom bulk image to make the same size as NF
    if im_pl.shape != im_sp.shape:
        size_ratio_h = im_pl.shape[0] / float(im_sp.shape[0])
        size_ratio_w = im_pl.shape[1] / float(im_sp.shape[1])

        x1 = int(round(x1 * size_ratio_w))
        x2 = int(round(x2 * size_ratio_w))
        y1 = int(round(y1 * size_ratio_h))
        y2 = int(round(y2 * size_ratio_h))

        # correct and crop plir image (using params from short pass)
        cropped_nf = nf_rotated[y1:y2, x1:x2]

        # upsize low res bulk
        tau_bulk_plir = ndimage.zoom(tau_bulk_plir, zoom=2.0, order=1)

        # make sure same size
        height = min(tau_bulk_plir.shape[0], cropped_nf.shape[0])
        width = min(tau_bulk_plir.shape[1], cropped_nf.shape[1])
        tau_bulk_plir = tau_bulk_plir[:height, :width]
        cropped_nf = cropped_nf[:height, :width]
        assert tau_bulk_plir.shape == cropped_nf.shape
    else:
        cropped_nf = nf_rotated[y1:y2, x1:x2]

    if False:
        view = ImageViewer(tau_bulk_plir)
        ImageViewer(cropped_nf)
        view.show()
        sys.exit()

    if parameters.PLIR_INTERPOLATE_MARKER_WIDTH > 0 and features['marker_loc'] > 0:
        # interpolate marker
        print features['marker_loc']
        locs = np.array([int(round(features['marker_loc']))], np.int32)
        cropped_nf = np.ascontiguousarray(cropped_nf, np.float32)
        pixel_ops.InterpolateBBs(cropped_nf, locs, parameters.PLIR_INTERPOLATE_MARKER_WIDTH)
        tau_bulk_plir = np.ascontiguousarray(tau_bulk_plir, np.float32)
        pixel_ops.InterpolateBBs(tau_bulk_plir, locs, parameters.PLIR_INTERPOLATE_MARKER_WIDTH)

        if False:
            view = ImageViewer(cropped_nf)
            ImageViewer(tau_bulk_plir)
            view.show()

    # correct for doping and transfer to bulk
    c_vals = fit_c_vals(cropped_nf, tau_bulk_plir, spline_plc)
    doping = ndimage.gaussian_filter1d(c_vals, sigma=2, mode="reflect")

    if False:
        ImageViewer(cropped_nf)
        plt.figure()
        plt.plot(c_vals)
        plt.plot(doping)
        plt.show()

    nf_dope = cropped_nf * np.r_[doping]
    nf_dope[nf_dope > spline_plc[0][-1]] = spline_plc[0][-1]

    tau_bulk_nf = interpolate.splev(nf_dope.flatten(), spline_plc).astype(np.float32).reshape(
        cropped_nf.shape)
    _, upper_p = ip.get_percentile(tau_bulk_nf, 0.0001)
    pixel_ops.ClipImage(tau_bulk_nf, 0.1, upper_p)

    features['_C_vals'] = doping.astype(np.float32)
    features['im_tau_bulk_f32'] = tau_bulk_nf
    features['im_tau_bulk_u8'] = (ip.scale_image(tau_bulk_nf) * 255).astype(np.uint8)
    features['im_cropped_nf_u8'] = (ip.scale_image(cropped_nf) * 255).astype(np.uint8)
    features['im_cropped_nf_u16'] = np.round(cropped_nf).astype(np.uint16)
    features['im_cropped_sp_u16'] = np.round(cropped_sp).astype(np.uint16)
    features['im_cropped_lp_u16'] = np.round(cropped_lp).astype(np.uint16)

    if False:
        print tau_bulk_nf.min(), tau_bulk_nf.max()
        _, upper_p = ip.get_percentile(tau_bulk_nf, 0.001)
        pixel_ops.ClipImage(tau_bulk_nf, 0.1, upper_p)
        # print interpolate.splev([0.0, 0.0245], spline_plc)
        #ImageViewer(cropped_nf * np.r_[doping])
        #ImageViewer(tau_bulk_nf)

        plt.figure()
        plt.plot(tau_bulk_plir.mean(axis=0))
        plt.plot(tau_bulk_nf.mean(axis=0))

        # plt.figure()
        # plt.hist(tau_bulk_full.flat, bins=100)
        if False:
            plt.figure()
            plt.plot(doping)
            plt.plot(c_vals)
            plt.figure()
            pl = np.mean(cropped_nf, axis=0)
            plt.plot(pl, label="PL")
            plt.legend()
        plt.show()

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start

    return True


def load_transfer(fn):
    with open(fn) as f:
        lines = [line.strip() for line in f if len(line.strip()) > 0][1:]
        vals = np.empty((len(lines), 5), np.float32)
        for e, line in enumerate(lines):
            vals[e, :] = np.array([float(v) for v in line.split('\t')], np.float32)

    return vals


def interpolate_transfer(vals, debug=False):
    tau = vals[:, 0]

    spline_plir = interpolate.splrep(vals[:, 1], tau, s=0)
    spline_nf = interpolate.splrep(vals[:, 2], tau, s=0)
    spline_sp = interpolate.splrep(vals[:, 3], tau, s=0)
    spline_lp = interpolate.splrep(vals[:, 4], tau, s=0)

    if debug:
        if False:
            xs = np.linspace(vals[:, 3].min(), vals[:, 3].max(), 1000)
            ys = interpolate.splev(xs, spline_sp)
            label = "sp"
        else:
            xs = np.linspace(vals[:, 1].min(), vals[:, 1].max(), 1000)
            ys = interpolate.splev(xs, spline_plir)
            label = "ratio"

        plt.figure()
        #plt.plot(vals[:, 1], tau, 'o')
        plt.plot(xs, ys, linewidth=2)
        plt.xlabel(label)
        plt.ylabel('Bulk lifetime')
        plt.show()

    return spline_plir, spline_nf, spline_sp, spline_lp


def main():
    pass


if __name__ == '__main__':
    main()
