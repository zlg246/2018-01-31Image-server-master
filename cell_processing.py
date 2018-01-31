import math
import math
import numpy as np
from scipy import ndimage, optimize, interpolate
import parameters
import matplotlib.pylab as plt
import sys
from image_processing import ImageViewer
import cv2
import pixel_ops
import image_processing as ip
import features_cz_wafer as cz_wafer
import glob
from skimage.transform import probabilistic_hough_line as probabilistic_hough
from skimage import draw
import itertools
from skimage import measure
import collections


class CellTemplateException(Exception):
    pass


class CellFingersException(Exception):
    pass


class MissingBusbarsException(Exception):
    pass


def register(im1, im2):
    if False:
        def Dist(params):
            tx, ty = params
            sy = 1
            M = np.float32([[1, 0, tx], [0, sy, ty]])
            im2_reg = cv2.warpAffine(im2, M, (im2.shape[1], im2.shape[0]))
            return np.power(im2_reg - im1, 2).mean()

        params_op = optimize.fmin_powell(Dist, (0, 0), ftol=1.0, disp=False)
        tx, ty = params_op
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        im2_reg = cv2.warpAffine(im2, M, (im2.shape[1], im2.shape[0]))
    else:
        h, w = im1.shape

        def Dist(params):
            tx, ty, r = params
            M = cv2.getRotationMatrix2D((w // 2, h // 2), r, 1.0)
            M[0, 2] += tx
            M[1, 2] += ty

            im2_reg = cv2.warpAffine(im2, M, (im2.shape[1], im2.shape[0]))
            # return np.power(im2_reg-im1, 2).mean()
            return np.abs(im2_reg - im1).mean()

        params_op = optimize.fmin_powell(Dist, (0, 0, 0), ftol=1.0, disp=False)
        tx, ty, r = params_op
        M = cv2.getRotationMatrix2D((w // 2, h // 2), r, 1.0)
        M[0, 2] += tx
        M[1, 2] += ty
        im2_reg = cv2.warpAffine(im2, M, (im2.shape[1], im2.shape[0]))

    if False:
        print np.power(im2_reg - im1, 2).mean()
        view = ImageViewer(im1)
        view = ImageViewer(im2_reg)
        view.show()
        sys.exit()

    return im2_reg, np.power(im2_reg - im1, 2).mean()


def find_busbars(im, features):
    h, w = im.shape

    if parameters.CELL_NO_BBS:
        features['bb_detection_mode'] = -1
        features['_busbar_cols'] = np.array([], np.int32)
        features['busbar_width'] = 0
        features['mask_busbar_edges'] = np.zeros_like(im, dtype=np.uint8)
        features['mask_busbar_filled'] = np.zeros_like(im, dtype=np.uint8)
        return

    ####################
    # BUSBAR LOCATIONS #
    ####################
    def pass_symmetry_test(bb_locs, debug=False):
        # check for symmetry in bb locations
        num_bb = len(bb_locs)
        if num_bb == 2:
            dist = abs(bb_locs[0] - (w - bb_locs[1]))
        elif num_bb == 3:
            dist1 = abs(w // 2 - bb_locs[1])
            dist2 = abs(bb_locs[0] - (w - bb_locs[2]))
            dist = (dist1 + dist2) / 2.0
        elif num_bb == 4:
            dist1 = abs(bb_locs[0] - (w - bb_locs[3]))
            dist2 = abs(bb_locs[1] - (w - bb_locs[2]))
            dist = (dist1 + dist2) / 2.0
        elif num_bb == 5:
            dist1 = abs(w // 2 - bb_locs[2])
            dist2 = abs(bb_locs[0] - (w - bb_locs[4]))
            dist3 = abs(bb_locs[1] - (w - bb_locs[3]))
            dist = (dist1 + dist2 + dist3) / 3.0
        else:
            dist = 100

        if num_bb >= 3:
            spacing = bb_locs[1:] - bb_locs[:-1]
            spacing_ratio = spacing.max() / float(spacing.min())
        else:
            spacing_ratio = 1

        if debug:
            print dist, spacing_ratio

        return (dist <= 15) and (spacing_ratio < 2)

    # attempt 1: columns without much variation
    features['bb_detection_mode'] = None
    col_var = ndimage.gaussian_filter1d(features['_col_var'], sigma=2, mode="constant")
    col_var = ndimage.gaussian_filter1d(col_var, sigma=1)
    dip_strength = np.minimum(np.roll(col_var, 10), np.roll(col_var, -10)) - col_var
    threshold = parameters.CELL_BB_MIN
    e = len(col_var) // 25
    col_var[:e] = 0
    col_var[-e:] = 0
    busbar_locations = np.where((col_var < threshold) &
                                (col_var < np.roll(col_var, 1)) &
                                (dip_strength > 0.07) &
                                (col_var < np.roll(col_var, -1)))[0]

    # ignore anything 0.2 higher than lowest
    # - for multi cells, impure areas can cause low variation, so skip this test
    if len(busbar_locations) > 0 and features['_alg_mode'] != 'multi cell':
        busbar_locations = busbar_locations[(col_var[busbar_locations] - col_var[busbar_locations].min()) < 0.2]
        busbar_locations.sort()

    # group anything very close together
    if len(busbar_locations) > 0:
        diffs = busbar_locations[1:] - busbar_locations[:-1]
        for i in np.where(diffs < 10)[0]:
            middle = (busbar_locations[i] + busbar_locations[i + 1]) // 2
            busbar_locations[i] = middle
            busbar_locations[i + 1] = -1
        busbar_locations = busbar_locations[busbar_locations >= 0]

    if False:
        print busbar_locations, pass_symmetry_test(busbar_locations, debug=True), parameters.CELL_BB_MIN

        ImageViewer(im)
        plt.figure()
        plt.plot(col_var)
        plt.plot(dip_strength)
        plt.vlines(busbar_locations, ymin=0, ymax=1)
        plt.show()
        # sys.exit()

    if pass_symmetry_test(busbar_locations):
        features['bb_detection_mode'] = 1

    if features['bb_detection_mode'] is None:
        # attempt 2: local minimums of intensity profile
        if im.shape[0] > 600:
            sigma = 5
            offset = 10
        else:
            sigma = 2.5
            offset = 5
        col_mean = ndimage.gaussian_filter1d(im.mean(axis=0), sigma=sigma, mode="constant")
        col_mean /= col_mean.max()
        dip_strength = np.minimum(np.roll(col_mean, offset), np.roll(col_mean, -offset)) - col_mean
        dip_strength[dip_strength < 0] = 0
        p10 = int(len(dip_strength) * 0.09)
        dip_strength[:p10] = 0
        dip_strength[-p10:] = 0

        dip_thresh = dip_strength.std() * parameters.CELL_BB_MODE_2_STD

        busbar_locations = np.where((col_mean < np.roll(col_mean, 1)) &
                                    (col_mean < np.roll(col_mean, -1)) &
                                    (((dip_strength > 0.05) & (col_mean < 0.6)) | (dip_strength > dip_thresh)))[0]

        # ignore anything too close to edge
        busbar_locations = busbar_locations[busbar_locations > 20]
        busbar_locations = busbar_locations[busbar_locations < w - 20]
        busbar_locations.sort()

        if False:
            print busbar_locations, dip_thresh, pass_symmetry_test(busbar_locations)
            ImageViewer(im)
            plt.figure()
            plt.plot(col_mean)
            plt.plot(dip_strength)
            plt.plot(busbar_locations, col_mean[busbar_locations], 'o')
            plt.show()
            # sys.exit()

        if pass_symmetry_test(busbar_locations):
            features['bb_detection_mode'] = 2

    if features['bb_detection_mode'] is None:
        # attempt 3:  gradient of intensity profile
        dips = np.abs(np.minimum(np.roll(col_mean, shift=15), np.roll(col_mean, shift=-15)) - col_mean)
        e = len(col_mean) // 10
        dips[:e] = 0
        dips[-e:] = 0

        # find strongest dips
        local_maxs = np.where(np.logical_and(dips > np.roll(dips, shift=1),
                                             dips > np.roll(dips, shift=-1)))[0]

        # if dip near middle -> 3 busbars, else 2
        w2 = len(dips) // 2
        if dips[w2 - 10:w2 + 10].max() > 0.02:
            num_bb = 3
        else:
            num_bb = 2
        busbar_locations = local_maxs[np.argsort(dips[local_maxs])[-num_bb:]]
        busbar_locations.sort()

        if False:
            print busbar_locations
            ImageViewer(im)
            plt.figure()
            plt.plot(col_mean)
            plt.plot(dips)
            plt.plot(busbar_locations, col_mean[busbar_locations], 'o')
            plt.show()
            sys.exit()

        if pass_symmetry_test(busbar_locations):
            features['bb_detection_mode'] = 3

    if features['bb_detection_mode'] is None:
        # attempt 4
        # - peaks in interior of
        e = len(col_mean) // 10
        peaks = col_mean - np.maximum(np.roll(col_mean, shift=15), np.roll(col_mean, shift=-15))
        local_maxs = np.where(np.logical_and(peaks > np.roll(peaks, shift=1),
                                             peaks > np.roll(peaks, shift=-1)))[0]
        local_maxs = local_maxs[local_maxs > e]
        local_maxs = local_maxs[local_maxs < len(col_mean) - e]
        busbar_locations = local_maxs[peaks[local_maxs] > 0.04]

        if False:
            print busbar_locations
            ImageViewer(im)
            plt.figure()
            plt.plot(peaks)
            plt.plot(local_maxs, peaks[local_maxs], 'o')
            plt.figure()
            plt.plot(col_mean)
            plt.plot(busbar_locations, col_mean[busbar_locations], 'o')
            plt.show()

        if pass_symmetry_test(busbar_locations):
            features['bb_detection_mode'] = 4

    if features['bb_detection_mode'] is None:
        if False:
            plt.figure()
            plt.plot(im.mean(axis=0))
            plt.show()

        raise MissingBusbarsException

    if False:
        print "Number of busbars: %d" % (len(busbar_locations))
        print features['bb_detection_mode']
        mask = np.zeros_like(im, np.uint8)
        mask[:, busbar_locations] = 1
        rgb = ip.overlay_mask(im, mask)
        view = ImageViewer(rgb)
        view.show()

    # fine tune: see if we get a better fit by shifting left or right
    bw_max = int(w * 0.025)
    bl = busbar_locations[0]
    bb_profile = np.median(im[:, bl - bw_max:bl + bw_max + 1], axis=0)
    bb_profile -= bb_profile.min()
    bb_profile /= bb_profile.max()
    for bb in range(1, len(busbar_locations)):
        bl = busbar_locations[bb]
        bb_profileX = np.median(im[:, bl - bw_max:bl + bw_max + 1], axis=0)
        bb_profileX -= bb_profileX.min()
        bb_profileX /= bb_profileX.max()

        # find offset
        shifts = np.arange(-5, 6)
        differences = [np.abs(bb_profile - np.roll(bb_profileX, s)).mean() for s in shifts]
        best_shift = shifts[np.argmin(differences)]
        busbar_locations[bb] -= best_shift

        if False:
            print best_shift
            plt.figure()
            plt.plot(shifts, differences)
            plt.figure()
            plt.plot(bb_profile)
            plt.plot(bb_profileX)
            plt.plot(np.roll(bb_profileX, best_shift), '--')
            plt.show()

    features['_busbar_cols'] = busbar_locations

    ################
    # BUSBAR WIDTH #
    ################

    # create median busbar
    bb_stack = np.empty((h, bw_max * 2 + 1, len(busbar_locations)), np.float32)
    for e, bl in enumerate(busbar_locations):
        bb_stack[:, :, e] = im[:, bl - bw_max:bl + bw_max + 1]
    bb_template = np.median(bb_stack, axis=2)
    features['_bb_template'] = bb_template

    if False:
        plt.figure()
        for e, bl in enumerate(busbar_locations):
            plt.plot(im[:, bl - bw_max:bl + bw_max + 1].mean(axis=0))

        view = ImageViewer(bb_stack[:, :, 2])
        ImageViewer(bb_template)
        ImageViewer(im)
        view.show()
        sys.exit()

    if int(parameters.CELL_BB_WIDTH_MODE) == 1:
        # base on column variation
        im_peaks = bb_template[features['_peak_row_nums'], :]
        im_fingers = bb_template[features['_finger_row_nums'][:-1], :]
        diff = (im_peaks - im_fingers)
        col_var = np.median(diff, axis=0)  # diff.mean(axis=0)
        col_var /= min(col_var[0], col_var[-1])

        low_var = np.where(col_var < 0.8)[0]
        if len(low_var) < 2:
            raise MissingBusbarsException
        left, right = np.where(col_var < 0.8)[0][[0, -1]]

        if False:
            print col_var[(left + right) // 2]
            ImageViewer(diff)
            ImageViewer(bb_template)
            plt.figure()
            plt.plot(col_var)
            plt.vlines([left, right], 0, 1)
            plt.show()

        if col_var[(left + right) // 2] >= 0.8:
            raise MissingBusbarsException
    elif int(parameters.CELL_BB_WIDTH_MODE) == 2:
        # base on column intensity
        cols = bb_template.mean(axis=0)
        cols /= cols.max()
        dark_area = np.where(cols < parameters.CELL_BB_THRESH)[0]
        if len(dark_area) < 5:
            raise MissingBusbarsException
        left, right = dark_area[[0, -1]]

        if False:
            # ImageViewer(diff)
            ImageViewer(bb_template)
            plt.figure()
            plt.plot(cols)
            plt.vlines([left, right], 0, 1)
            plt.show()
    elif int(parameters.CELL_BB_WIDTH_MODE) == 3:
        left = (bb_template.shape[1] // 2) - parameters.CELL_BB_WIDTH
        right = (bb_template.shape[1] // 2) + parameters.CELL_BB_WIDTH

        if False:
            print left, right
            cols = bb_template.mean(axis=0)
            cols /= cols.max()
            ImageViewer(bb_template)
            plt.figure()
            plt.plot(cols)
            plt.vlines([left, right], 0, 1)
            plt.show()
    else:
        print "ERROR: Unknown busbar detection mode"

    features['busbar_width'] = right - left

    # create a mask, and use it to insert 1's for H&V filtering
    busbar_mask = np.zeros_like(im, dtype=np.uint8)
    busbar_filled = np.zeros_like(im, dtype=np.bool)
    for e, bl in enumerate(busbar_locations):
        busbar_mask[:, bl - bw_max + left] = 1
        busbar_mask[:, bl - bw_max + right] = 1
        busbar_filled[:, bl - bw_max + left:bl - bw_max + right + 1] = True

    if False:
        print features['busbar_width']
        overlay = ip.overlay_mask(im, busbar_mask)
        view = ImageViewer(overlay)
        ImageViewer(busbar_filled)
        view.show()
        sys.exit()

    features['mask_busbar_edges'] = busbar_mask
    features['mask_busbar_filled'] = busbar_filled


def normalise(cropped, features):
    # normalize using median of mid-point between fingers
    #  - don't want to use percentiles, as bright fingers act as outliers
    #    and reduce contrast everywhere else
    if '_peak_row_nums' in features:
        foreground = cropped[features['_peak_row_nums'], :]
        fg_mask = features['bl_cropped_u8'][features['_peak_row_nums'], :]
        fg_vals = foreground[fg_mask == 0]
        fg_median = np.median(fg_vals)
        norm = cropped / fg_median
    else:
        f = {}
        ip.histogram_percentiles(cropped, f)
        norm = cropped / f['hist_percentile_99']

    features['im_norm'] = norm

    # create an 8-bit display version
    display = cropped / features['hist_percentile_99.9']
    pixel_ops.ApplyThresholdGT_F32(display, display, 1.0, 1.0)
    features['im_cropped_u8'] = np.round(display * 255).astype(np.uint8)


def remove_cell_template(norm, features):
    if features['_fingers_grid']:
        # finger grid
        rh = int(round(features['finger_period_row']))
        cw = int(round(features['finger_period_col']))
        no_fingers = ndimage.uniform_filter(norm, size=(rh, cw))

        features['im_no_fingers'] = no_fingers

        # remove busbars
        no_bbs = no_fingers.copy()
        pixel_ops.InterpolateBBs(no_bbs, np.array(features['_busbar_cols'], np.int32), features['busbar_width'] + 6)
        features['im_no_figners_bbs'] = no_bbs

        if False:
            view = ImageViewer(norm)
            ImageViewer(no_fingers)
            ImageViewer(no_bbs)
            view.show()
    else:
        # remove fingers
        f_len = features['finger_period']
        f = np.ones((int(round(f_len)), 1), np.float32) / f_len
        no_lines = ndimage.correlate(norm, f)

        # sharpen
        F_LEN2 = int(round(1.5 * f_len))
        f2 = np.ones((1, F_LEN2), np.float32) / F_LEN2
        filtered = ndimage.correlate(norm, f2)
        filtered[filtered < 0.1] = 1.0
        if False:
            view = ImageViewer(norm)
            ImageViewer(no_lines)
            ImageViewer(filtered)
            view.show()

        edges = norm / filtered
        no_fingers = edges * no_lines
        features['im_no_fingers'] = no_fingers

        if '_busbar_cols' in features:
            # remove busbars
            no_bbs = no_fingers.copy()
            pixel_ops.InterpolateBBs(no_bbs, np.array(features['_busbar_cols'], np.int32), features['busbar_width'] + 6)
        else:
            no_bbs = no_fingers

        features['im_no_figners_bbs'] = no_bbs

        if False:
            view = ImageViewer(norm)
            ImageViewer(no_fingers)
            ImageViewer(no_bbs)
            view.show()


def cell_edge_width(im, features):
    h, w = im.shape
    h2 = h // 2

    # look for frequency content at the frequency of the finger period
    if False:
        mid = im[h2 - 50:h2 + 50, :]
        period = int(round(features['finger_period']))
        period_avg = np.empty((period, im.shape[1]), np.float32)
        for offset in range(period):
            period_avg[offset, :] = mid[offset::period, :].mean(axis=0)
        col_var = period_avg.max(axis=0) - period_avg.min(axis=0)
    else:
        im_peaks = im[features['_peak_row_nums'], :]
        im_fingers = im[features['_finger_row_nums'][:-1], :]
        diff = (im_peaks - im_fingers)
        # col_var = diff.mean(axis=0)
        col_var = np.median(diff, axis=0)

        if False:
            view = ImageViewer(im_fingers)
            ImageViewer(im_peaks)
            ImageViewer(diff)
            view.show()

    col_var -= col_var.min()
    col_var /= col_var.max()
    interior = np.where(col_var > parameters.CELL_EDGE_STD_THRESH)[0]
    left, right = interior[[0, -1]]

    features['_col_var'] = col_var
    if features['_alg_mode'] == 'multi cell':
        # since one side might be impure (= low intensity & low variation) select the
        #  smaller of the two estimates
        edge_width = max(1, min(left, w - right))
        left, right = edge_width, w - edge_width
        features['cell_edge_left'] = left
        features['cell_edge_right'] = right
        features['cell_edge_tb'] = edge_width
    else:
        features['cell_edge_left'] = max(left, 1)
        features['cell_edge_right'] = min(w - 1, right)
        features['cell_edge_tb'] = ((w - right) + left) // 2

    if False:
        print left, (w - right)
        # print features['cell_edge_width']
        plt.figure()
        plt.plot(col_var)
        plt.vlines([left, right], 0, col_var.max())
        view = ImageViewer(im)
        view.show()
        sys.exit()


def stripe_structure(cropped, features):
    features['bl_cropped_u8'] = np.zeros_like(cropped, np.uint8)
    return


def create_cell_mask(norm, features):
    # (1=background, 2=busbar, 4=finger, 8=cell edge)
    cell_mask = features['mask_busbar_filled'].astype(np.uint8) * 2
    cell_mask[features['_finger_row_nums'], :] += 4
    if 'mask_grid_cols' in features:
        cell_mask[:, features['mask_grid_cols']] |= 4

    r = features['im_center_dist_im']
    h, w = norm.shape
    left = features['cell_edge_left']
    right = features['cell_edge_right']
    top = features['cell_edge_tb']
    bottom = h - features['cell_edge_tb']
    cell_mask[:top, :] = 8
    cell_mask[bottom:, :] = 8
    cell_mask[:, :left] = 8
    cell_mask[:, right:] = 8
    edge_width = int((left + (w - right) + top + (h - bottom)) / 4.0) + 1
    features['cell_edge_width_avg'] = edge_width
    # pixel_ops.ApplyThresholdGT_F32_U8(r, cell_mask, features['wafer_radius']-edge_width, 8)
    pixel_ops.ApplyThresholdGT_F32_U8(r, cell_mask, features['wafer_radius'] - edge_width, 8)
    pixel_ops.ApplyThresholdGT_F32_U8(r, cell_mask, features['wafer_radius'], 1)
    features['bl_cropped_u8'] = cell_mask


def cell_structure(cropped, features):
    # create a map of distance from each pixel to center
    r, theta = np.empty_like(cropped, np.float32), np.empty_like(cropped, np.float32)
    pixel_ops.CenterDistance(r, theta, features['wafer_middle_y'], features['wafer_middle_x'])
    features['im_center_dist_im'] = r
    features['im_center_theta_im'] = theta
    if False:
        # in multi mode, don't estimate radius
        features['wafer_radius'] = features['_cell_diag'] + 1
    else:
        features['im_center_dist_rot'] = r
        features['im_center_theta_rot'] = theta

    # determine properties of the cell pattern
    if features['_fingers_grid']:
        find_fingers_perc(cropped, features)
    else:
        find_fingers(cropped, features)
    cell_edge_width(cropped, features)
    find_busbars(cropped, features)
    create_cell_mask(cropped, features)


def finger_profile(norm, bb_locs, features):
    # use super resolution to find the shape of fingers

    # find mid-points between busbars
    x1 = (bb_locs[0] + bb_locs[1]) // 2
    profile = norm[h // 4:-h // 4, x1 - 5:x1 + 6].mean(axis=1)
    peaks = np.where((profile > np.roll(profile, 1)) &
                     (profile > np.roll(profile, -1)))[0]
    peaks = peaks[peaks > features['finger_period']]
    peaks = peaks[peaks < profile.shape - features['finger_period']]

    # create a version with 10x resolutiobn
    f_super = interpolate.interp1d(np.arange(profile.shape[0]), profile)

    # for each peak, interpolate a better peak
    # add to super-version
    peaks_interp = []
    fp2 = (features['finger_period'] // 2) + 1
    # peaks_val = []
    S = 51
    finger_super = np.zeros(S, dtype=np.float32)
    for p in peaks:
        f = interpolate.interp1d([p - 1, p, p + 1], -1 * profile[p - 1:p + 2], kind="quadratic",
                                 bounds_error=False, fill_value=1.0)
        pi = optimize.fmin(f, p, disp=False)[0]
        peaks_interp.append(pi)
        finger_super += f_super(np.linspace(pi - fp2, pi + fp2, num=S))

        # peaks_val.append(-1*f(pi))

    finger_super /= len(peaks)

    if False:
        plt.figure()
        plt.plot(finger_super)
        plt.figure()
        plt.plot(profile)
        # plt.plot(peaks_interp, peaks_val, 'o')
        plt.show()


def find_fingers(im, features):
    """
    Find the period and locations of the grid fingers
    """
    row_profile = np.mean(im, axis=1)

    # if you smooth too much you lose fingers
    row_profile = ndimage.gaussian_filter1d(row_profile, sigma=0.5)
    row_profile /= row_profile.max()

    if False:
        view = ImageViewer(im)
        plt.figure()
        plt.plot(row_profile)
        plt.show()
        sys.exit()

    # find the local minimums
    grid_rows_mask = np.logical_and(row_profile < np.roll(row_profile, 1),
                                    row_profile < np.roll(row_profile, -1))
    grid_rows_mask[[0, -1]] = False
    peaks = np.where(row_profile > 0.2)[0]
    start, stop = peaks[0], peaks[-1]

    # make sure we don't have any fingers inside the cell edges
    if 'cell_edge_tb' in features:
        start = max(features['cell_edge_tb'], start)
        stop = min(stop, im.shape[0] - features['cell_edge_tb'])

    grid_rows_mask[:start] = False
    grid_rows_mask[stop:] = False
    grid_rows = np.where(grid_rows_mask)[0]

    # compute the period
    local_mins = grid_rows[5:-5]
    if len(local_mins) == 0:
        raise CellFingersException

    period = (local_mins[-1] - local_mins[0]) / float(len(local_mins) - 1)
    # features['mask_grid_rows'] = grid_rows_mask
    features['_finger_row_nums'] = grid_rows
    features['finger_period'] = period
    features['_peak_row_nums'] = (grid_rows[:-1] + (period // 2)).astype(np.int32)

    if False:
        # finger spacing analysis
        grid_rows_interpolated = []
        for i in range(1, len(grid_rows) - 1):
            x = grid_rows[i]
            xs = [x - 1, x, x + 1]
            ys = [row_profile[j] for j in xs]

            f = interpolate.interp1d(xs, ys, kind='quadratic', bounds_error=False, fill_value=max(ys))
            grid_rows_interpolated.append(optimize.fmin(f, x, disp=False)[0])
            if False:
                print grid_rows_interpolated[-1], f(grid_rows_interpolated[-1])
                xnew = np.linspace(xs[0], xs[-1], num=100, endpoint=True)
                ynew = f(xnew)

                plt.figure()
                plt.plot(xs, ys)
                plt.plot(xnew, ynew)
                plt.plot(grid_rows_interpolated[-1], f(grid_rows_interpolated[-1]), 'o')
                plt.show()

        grid_rows_interpolated = np.array(grid_rows_interpolated)
        diffs = grid_rows_interpolated[1:] - grid_rows_interpolated[:-1]

        xs = grid_rows[1:-1][1:] - (period / 2.0)

        f = np.poly1d(np.polyfit(xs, diffs, deg=2))
        xnew = np.linspace(xs[0], xs[-1], num=100, endpoint=True)

        plt.figure()
        plt.plot(xs, diffs)
        plt.plot(xnew, f(xnew))
        plt.show()

    if False:
        print "Period: {}".format(features['finger_period'])
        ImageViewer(im)

        plt.figure()
        plt.plot(grid_rows, row_profile[grid_rows], 'o')
        plt.plot(features['_peak_row_nums'], row_profile[features['_peak_row_nums']], 'o')
        plt.plot(row_profile)
        # if False: plt.vlines([grid_rows[0]+(i*features['finger_period']) for i in range(len(grid_rows))], row_profile.min(), row_profile.max())
        # else: plt.vlines(grid_rows_interpolated, row_profile.min(), row_profile.max())
        plt.show()
        sys.exit()


def find_fingers_perc(im, features):
    """
    Find the period and locations of the grid fingers
    """

    # rows
    row_profile = np.median(im, axis=1)
    row_profile /= row_profile.max()
    grid_rows_mask = np.logical_and(row_profile < np.roll(row_profile, 1),
                                    row_profile < np.roll(row_profile, -1))
    peaks = np.where(row_profile > 0.2)[0]
    start, stop = peaks[0], peaks[-1]
    grid_rows_mask[:start] = False
    grid_rows_mask[stop:] = False
    grid_rows = np.where(grid_rows_mask)[0]

    # compute the period
    local_mins = grid_rows[5:-5]
    if len(local_mins) == 0:
        raise CellTemplateException

    period = (local_mins[-1] - local_mins[0]) / float(len(local_mins) - 1)
    features['mask_grid_rows'] = grid_rows_mask
    features['_finger_row_nums'] = grid_rows
    features['finger_period_row'] = period
    features['_peak_row_nums'] = (grid_rows[:-1] + (period // 2)).astype(np.int32)

    # cols
    col_profile = np.median(im, axis=0)
    col_profile /= col_profile.max()
    grid_cols_mask = ((col_profile < np.roll(col_profile, 1)) &
                      (col_profile < np.roll(col_profile, -1)) &
                      (col_profile > 0.3))
    peaks = np.where(col_profile > 0.2)[0]
    start, stop = peaks[0], peaks[-1]
    grid_cols_mask[:start] = False
    grid_cols_mask[stop:] = False
    grid_cols = np.where(grid_cols_mask)[0]

    # compute the period
    local_mins = grid_cols[5:-5]
    if len(local_mins) == 0:
        raise CellTemplateException

    period = (local_mins[-1] - local_mins[0]) / float(len(local_mins) - 1)
    features['mask_grid_cols'] = grid_cols_mask
    features['_finger_col_nums'] = grid_cols
    features['finger_period_col'] = period

    if False:
        print "Row period: {}".format(features['finger_period_row'])
        print "Col period: {}".format(features['finger_period_col'])
        ImageViewer(im)

        mask = np.zeros_like(im, np.uint8)
        mask[:, grid_cols_mask] = 1
        mask[grid_rows_mask, :] = 1
        rgb = ip.overlay_mask(im, mask)
        ImageViewer(rgb)

        plt.figure()
        plt.plot(grid_rows, row_profile[grid_rows], 'o')
        plt.plot(row_profile)
        plt.figure()
        plt.plot(grid_cols, col_profile[grid_cols], 'o')
        plt.plot(col_profile)
        plt.show()
        sys.exit()


# @profile
def filter_v(filtered, features):
    h, w = filtered.shape

    # Vertical filtering: compare pixel to corresponding locations from grid
    #  rows above and below
    period = int(round(features['finger_period']))
    row_locs = features['_finger_row_nums']

    # can't filter at the top and bottom. the cutoff depends on top/bottom
    #  finger and distance to image edge
    middle_start = max(row_locs[0] + period, 2 * period)
    middle_stop = min(row_locs[-1] - period, h - (2 * period))
    rows_base = np.arange(0, h, period)
    filtered_v = filtered.copy()
    for s in range(period):
        rows = rows_base + s
        rows = rows[rows < h]
        if False:
            rows_im = np.ascontiguousarray(filtered[rows, :])
        else:
            rows_im = np.empty((len(rows), w), np.float32)
            pixel_ops.CopyRows(filtered, rows, rows_im)
        rows_filtered = np.zeros_like(rows_im)
        pixel_ops.FilterV(rows_im, rows_filtered)

        if False:
            print s
            print rows
            # view = ImageViewer(filtered)
            view = ImageViewer(rows_im)
            ImageViewer(rows_filtered)
            view.show()
            # sys.exit()

        # skip anything too close to top or bottom
        row_mask = ((rows > middle_start) & (rows < middle_stop - 1))
        # TODO: speedup the follownig
        filtered_v[rows[row_mask], :] = rows_filtered[row_mask]

    # can't filter areas near corners for the same reason
    if 'param_multi_wafer' not in features or not features['param_multi_wafer']:
        mask = features['im_center_dist'] > features['cell_radius'] - middle_start
        filtered_v[mask] = filtered[mask]

    if False:
        view = ImageViewer(filtered)
        view = ImageViewer(filtered_v)
        view.show()
        sys.exit()

    return filtered_v


def filter_h(filtered_v, features):
    #  - set busbars to high value so that adjoining defects are detected
    filered_filled = filtered_v.copy()
    if False:
        filered_filled[features['mask_busbar_filled']] = 255
        m = np.ones((1, 31), np.uint8)
        filtered_h = filter.rank.median(filered_filled, m)
    else:
        filered_filled[features['mask_busbar_filled']] = 1.0
        filtered_h = filered_filled.copy()
        pixel_ops.FilterH(filered_filled, filtered_h, parameters.F_LEN_H)

    if False:
        view = ImageViewer(filtered_v)
        ImageViewer(filered_filled)
        ImageViewer(filtered_h)
        view.show()
        sys.exit()

    return filtered_h


def multi_cracks(features):
    im = features['im_no_figners_bbs']
    h, w = im.shape
    kernel = np.ones((5, 5), np.uint8)

    # apply anisotropic diffusion (parameters need to be set properly).
    im_adf = ip.anisotropic_diffusion(im.copy(),parameters.ANISO_DIFF_NUM_ITERATION, parameters.ANISO_DIFF_KAPPA, parameters.ANISO_DIFF_GAMMA)
    im_diff = im_adf - im

    # binarize image using intensity mean & standard deviation.
    bin_threshold = np.mean(im_diff) + parameters.IMG_BIN_INTENSITY_FACTOR * np.std(im_diff)
    im_bin = (im_diff > bin_threshold).astype(np.uint8)

    if False:
        view = ImageViewer(im)
        ImageViewer(im_adf)
        ImageViewer(im_diff)
        ImageViewer(im_bin)
        view.show()

    # get connected regions
    ccs, num_ccs = ip.connected_components(im_bin)

    # find connected components big enough to consider for further analysis
    min_area = im.shape[0] / parameters.CONN_REGION_MIN_SIZE_FACTOR
    cc_sizes = np.zeros(num_ccs + 1, np.int32)
    pixel_ops.CCSizes(ccs, cc_sizes)
    big_ccs = np.where(cc_sizes > min_area)[0]

    if False:
        print big_ccs
        view = ImageViewer(im)
        ImageViewer(ccs)
        view.show()

    mask_cracks = np.zeros_like(im, np.uint8)
    crack_count = 0
    for cc_label in big_ccs:
        ys, xs = np.where(ccs == cc_label)

        mean_intensity = im[ys, xs].mean()
        if mean_intensity > parameters.CONN_REGION_MIN_MEAN_INTENSITY:
            continue

        # width & height
        crack_width = xs.max() - xs.min()
        crack_height = ys.max() - ys.min()

        # remove vertical or horizontal lines.
        if crack_width < parameters.CONN_REGION_MIN_WID_HEI or crack_height < parameters.CONN_REGION_MIN_WID_HEI:
            continue

        # crop crack
        window_y1, window_x1 = max(0, ys.min() - 2), max(0, xs.min() - 2)
        window_y2, window_x2 = min(h, ys.max() + 3), min(w, xs.max() + 3)
        im_window = im[window_y1:window_y2, window_x1:window_x2]
        crack_im = ccs[window_y1:window_y2, window_x1:window_x2] == cc_label

        # compute region properties for big enough cracks (this is expensive, so only do if high prob cracks)
        region = measure.regionprops(crack_im.astype(np.uint8))[0]
        if region.eccentricity < parameters.CONN_REGION_MAX_ECCENTRICITY:
            continue

        # determine min x or y coordinates.
        reg_coords = region.coords
        if crack_width < crack_height:  # vertical lines.
            x_min_idx = np.argmin(reg_coords[:, 0])
            xy_min = reg_coords[x_min_idx, :]
        else:
            y_min_idx = np.argmin(reg_coords[:, 1])
            xy_min = reg_coords[y_min_idx, :]
        relative_coords = reg_coords - xy_min
        relative_dis = np.sqrt(np.sum(relative_coords ** 2., axis=1))
        relative_dis = np.round(relative_dis).astype(np.int32)
        dis_freq = collections.Counter(relative_dis).values()

        # get standard deviation of distances.
        dis_std = np.std(dis_freq)

        # calculate intensity contrast to neighbouring region.
        candidate_dilate = cv2.dilate(crack_im.astype(np.uint8), kernel, iterations=1)
        candidate_outline = candidate_dilate - crack_im
        surrounding_mean = im_window[candidate_outline == 1].mean()

        # intensity difference with neighbouring region.
        inten_diff = surrounding_mean - mean_intensity

        # intensity ratio with neighbouring region.
        inten_ratio = inten_diff / mean_intensity

        # enforce conditions to detect cracks (parameters need to be set properly).
        if inten_diff < parameters.CONN_REGION_MAX_INTENSITY_DIFF or inten_ratio < parameters.CONN_REGION_MAX_INTENSITY_RATIO or \
                        surrounding_mean < parameters.SURR_REGION_MAX_MEAN_INTENSITY or dis_std > parameters.CONN_REGION_MIN_DISTANCE_STD:
            continue

        if False:
            view = ImageViewer(crack_im)
            ImageViewer(candidate_outline)
            ImageViewer(im_window)
            view.show()

        # keep cracks that have passed the tests & compute properties
        crack_count += 1
        mask_cracks[ys, xs] = cz_wafer.DEFECT_CRACK

        if 'input_param_verbose' not in features or features['input_param_verbose'] or crack_count < 6:
            cz_wafer.crack_properties(ys, xs, crack_count, features, mask_cracks)

        if crack_count >= parameters.MAX_NUM_CRACKS:
            break

    if False:
        view = ImageViewer(im)
        # ImageViewer(label_im)
        ImageViewer(ccs)
        ImageViewer(im_bin)
        view.show()

    features['mk_cracks_u8'] = mask_cracks
    features['defect_count'] = crack_count
    if crack_count > 0:
        features['defect_present'] = 1
    else:
        features['defect_present'] = 0

    # thin before finding length
    crack_skel = mask_cracks.copy()
    ys, xs = np.where(mask_cracks)
    pixel_ops.FastThin(crack_skel, ys.copy().astype(np.int32), xs.copy().astype(np.int32), ip.thinning_lut)
    features['defect_length'] = crack_skel.sum()
    features['_crack_skel'] = crack_skel


def mono_cracks(features):
    struct = ndimage.generate_binary_structure(2, 1)
    im = features['im_norm']
    im_no_rows = features['im_no_fingers']
    h, w = im.shape

    # apply a filter that enhances thin dark lines
    smoothed = cv2.GaussianBlur(im_no_rows, ksize=(0, 0), sigmaX=1.0, borderType=cv2.BORDER_REPLICATE)
    dark_lines = np.zeros(smoothed.shape, np.float32)
    pixel_ops.CrackEnhance2(smoothed, dark_lines)

    if False:
        view = ImageViewer(im_no_rows)
        ImageViewer(smoothed)
        ImageViewer(dark_lines)
        view.show()
        sys.exit()

    # ignore background
    r = features['wafer_radius']
    pixel_ops.ApplyThresholdGT_F32(features['im_center_dist_im'], dark_lines, r, 0)

    # HOUGH PARAMS
    LINE_THRESH = 0.07
    MIN_LINE_LEN = 0.017  # 0.025
    LINE_GAP = 4
    ANGLE_TOL = 15
    dark_lines_binary = (dark_lines > LINE_THRESH).astype(np.uint8)
    line_length = int(round(im.shape[0] * MIN_LINE_LEN))
    lines = probabilistic_hough(dark_lines_binary, threshold=20, line_length=line_length, line_gap=LINE_GAP,
                                theta=np.deg2rad(np.r_[np.arange(45 - ANGLE_TOL, 45 + ANGLE_TOL + 1),
                                                       np.arange(135 - ANGLE_TOL, 135 + ANGLE_TOL + 1)]))

    line_im = np.zeros_like(dark_lines)
    edge_dist = int(round(im.shape[0] * 0.035))
    coms = []
    middle_y, middle_x = features['wafer_middle_y'], features['wafer_middle_x']
    for line in lines:
        r0, c0, r1, c1 = line[0][1], line[0][0], line[1][1], line[1][0]
        rs, cs = draw.line(r0, c0, r1, c1)
        line_im[rs, cs] = 1
        coms.append(cs.mean())

        # connect to edge
        # first end
        rs_end1, cs_end1 = rs[:edge_dist], cs[:edge_dist]
        rs_ex1 = rs_end1 + (rs_end1[0] - rs_end1[-1])
        cs_ex1 = cs_end1 + (cs_end1[0] - cs_end1[-1])
        center_dists = np.sqrt((cs_ex1 - middle_x) ** 2 + (rs_ex1 - middle_y) ** 2)
        mask = ((rs_ex1 >= 0) & (rs_ex1 < h) &
                (cs_ex1 >= 0) & (cs_ex1 < w) &
                (center_dists < r))
        # make sure some pixels have been mask (i.e. outside of cell) and is dark or short
        # print im_no_rows[rs_ex1[mask], cs_ex1[mask]].mean(), mask.sum()
        if mask.sum() < len(rs_ex1) and (im_no_rows[rs_ex1[mask], cs_ex1[mask]].mean() < 0.45 or
                                                 mask.sum() < 9):
            line_im[rs_ex1[mask], cs_ex1[mask]] = 2

        # second end
        rs_end2, cs_end2 = rs[-edge_dist:], cs[-edge_dist:]
        rs_ex2 = rs_end2 + (rs_end2[-1] - rs_end2[0])
        cs_ex2 = cs_end2 + (cs_end2[-1] - cs_end2[0])
        center_dists = np.sqrt((cs_ex2 - middle_x) ** 2 + (rs_ex2 - middle_y) ** 2)
        mask = ((rs_ex2 >= 0) & (rs_ex2 < h) &
                (cs_ex2 >= 0) & (cs_ex2 < w) &
                (center_dists < r))
        # make sure some pixels have been mask (i.e. outside of cell) and is dark or short
        if mask.sum() < len(cs_ex2) and (im_no_rows[rs_ex2[mask], cs_ex2[mask]].mean() < 0.45 or
                                                 mask.sum() < 9):
            line_im[rs_ex2[mask], cs_ex2[mask]] = 2

    # join cracks that straddle BBs
    bb_cols = np.r_[0, features['_busbar_cols'], w - 1]
    for i1, i2 in itertools.combinations(range(len(lines)), 2):
        c1 = min(coms[i1], coms[i2])
        c2 = max(coms[i1], coms[i2])
        # make sure on different sides of BB (compare midpoints)
        straddle = False
        for bb in range(len(bb_cols) - 2):
            if bb_cols[bb] < c1 < bb_cols[bb + 1] < c2 < bb_cols[bb + 2]:
                straddle = True
                break
        if not straddle:
            continue

        # make sure similar orientation & offset
        def orientation(r0, c0, r1, c1):
            orien = math.degrees(math.atan2(r1 - r0, c1 - c0))
            if orien < 0:  orien += 180
            if orien > 180: orien -= 180
            return orien

        or1 = orientation(lines[i1][0][1], lines[i1][0][0], lines[i1][1][1], lines[i1][1][0])
        or2 = orientation(lines[i2][0][1], lines[i2][0][0], lines[i2][1][1], lines[i2][1][0])
        or_diff = abs(or2 - or1)
        if or_diff > 5:
            continue

        # find line between closest points
        if coms[i1] < coms[i2]:
            line1, line2 = lines[i1], lines[i2]
        else:
            line1, line2 = lines[i2], lines[i1]
        joining_line = draw.line(line1[1][1], line1[1][0], line2[0][1], line2[0][0])
        if len(joining_line[0]) > 0.05 * w:
            continue
        line_im[joining_line] = 3

    if False:
        view = ImageViewer(im_no_rows)
        ImageViewer(dark_lines)
        ImageViewer(dark_lines_binary)
        ImageViewer(line_im)
        view.show()
        sys.exit()

    # clean up lines
    line_im = ndimage.binary_closing(line_im, struct, iterations=2).astype(np.uint8)
    ys, xs = np.where(line_im)
    pixel_ops.FastThin(line_im, ys.copy().astype(np.int32), xs.copy().astype(np.int32), ip.thinning_lut)
    line_im = ndimage.binary_dilation(line_im, struct)

    # filter by "strength", which is a combination of darkness and length
    ccs, num_ccs = ip.connected_components(line_im)
    pixel_ops.ApplyThresholdGT_F32(dark_lines, dark_lines, 0.3, 0.3)
    if False:
        strength = ndimage.sum(dark_lines, labels=ccs, index=np.arange(num_ccs + 1))
    else:
        # median will be more robust than mean (dark spots can lead to false positives)
        median_vals = ndimage.median(dark_lines, labels=ccs, index=np.arange(num_ccs + 1))
        lengths = np.zeros(num_ccs + 1, np.int32)
        pixel_ops.CCSizes(ccs, lengths)
        strength = median_vals * lengths
    strength[0] = 0
    strongest_candidates = np.argsort(strength)[::-1]
    strongest_candidates = strongest_candidates[strength[strongest_candidates] > parameters.CELL_CRACK_STRENGTH]

    if False:
        # print strongest_candidates
        strength_map = np.take(strength, ccs)
        candidates = (strength_map > parameters.CELL_CRACK_STRENGTH).astype(np.uint8)
        view = ImageViewer(strength_map)
        ImageViewer(ip.overlay_mask(im, candidates, 'r'))
        view.show()
        sys.exit()

    # filter each candidate using other features
    mask_cracks = np.zeros_like(im, np.uint8)
    locs = ndimage.find_objects(ccs)
    crack_count = 0
    for cc_label in strongest_candidates:
        e = cc_label - 1
        y1, y2 = max(0, locs[e][0].start - 3), min(h, locs[e][0].stop + 3)
        x1, x2 = max(0, locs[e][1].start - 3), min(w, locs[e][1].stop + 3)
        crack = (ccs[y1:y2, x1:x2] == cc_label)
        im_win = im[y1:y2, x1:x2]
        ys, xs = np.where(crack)
        ys = ys + y1
        xs = xs + x1
        com_y = ys.mean()
        com_x = xs.mean()

        if False:
            view = ImageViewer(crack)
            ImageViewer(im_win)
            view.show()

        # remove cracks along corner edge by checking if center of mass
        #  is same distance from middle as cell radius, and parallel to edge
        center_dists = np.sqrt((ys - (h / 2.0)) ** 2 + (xs - (w / 2.0)) ** 2)
        center_dist = center_dists.mean()
        dist_range = center_dists.max() - center_dists.min()
        r = features['wafer_radius'] - features['cell_edge_tb']
        center_ratio = (min(center_dist, r) / max(center_dist, r))
        if center_ratio > 0.98 and dist_range < 10:
            continue

        # keep cracks that have passed the tests & compute properties
        crack_count += 1
        mask_cracks[y1:y2, x1:x2][crack] = cz_wafer.DEFECT_CRACK

        if ('input_param_verbose' not in features or features['input_param_verbose'] or crack_count < 6):
            cz_wafer.crack_properties(ys, xs, crack_count, features, mask_cracks)

        if crack_count >= parameters.MAX_NUM_CRACKS:
            break

    if False:
        # view = ImageViewer(ccs)
        print "Crack pixels: ", mask_cracks.sum()
        view = ImageViewer(ccs)
        ImageViewer(ip.overlay_mask(im, mask_cracks, 'r'))
        view.show()
        sys.exit()

    features['mk_cracks_u8'] = mask_cracks
    features['defect_count'] = crack_count
    if crack_count > 0:
        features['defect_present'] = 1
    else:
        features['defect_present'] = 0

    # thin before finding length
    crack_skel = mask_cracks.copy()
    ys, xs = np.where(mask_cracks)
    pixel_ops.FastThin(crack_skel, ys.copy().astype(np.int32), xs.copy().astype(np.int32), ip.thinning_lut)
    features['defect_length'] = crack_skel.sum()
    features['_crack_skel'] = crack_skel


def main():
    if False:
        import features_cz_cell
        features_cz_cell.main()
    elif False:
        import features_rs_cell
        features_rs_cell.main()
    elif True:
        import features_multi_cell
        features_multi_cell.main()
    else:
        import cropping
        cropping.main()


if __name__ == '__main__':
    main()
