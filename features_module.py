import sys
import numpy as np
import image_processing as ip
from image_processing import ImageViewer
import pixel_ops
import cv2
from scipy import ndimage, interpolate, optimize, signal
import matplotlib.pylab as plt
import matplotlib
import math
import multiprocessing as mp
import timeit
import string
import parameters
import features_multi_cell as multi
import cell_processing as cell
from statsmodels.tsa import stattools
import os
#matplotlib.style.use('ggplot')

# np.seterr(all='print')
np.seterr(all='raise')


class ModuleGridException(Exception):
    pass


def find_module_rotation(im, features):
    if False:
        view = ImageViewer(im)
        view.show()
        sys.exit()

    rotated = np.empty_like(im)
    h, w = im.shape
    max_r = 0.5
    crop = int(round(math.tan(math.radians(max_r)) * (im.shape[0] / 2.0)))
    rotations = np.linspace(-max_r, max_r, num=11)
    scores = []
    for rotation in rotations:
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        cv2.warpAffine(im, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, dst=rotated)

        cropped = rotated[crop:-crop, crop:-crop]
        # cols = cropped.mean(axis=0)
        # dog1 = ndimage.gaussian_filter1d(cols, sigma=10) - cols
        rows = cropped.mean(axis=1)
        dog2 = ndimage.gaussian_filter1d(rows, sigma=10) - rows

        # scores.append(dog1.std() + dog2.std())
        scores.append(dog2.std())

        if False:
            print dog2.std()
            # plt.figure()
            # plt.plot(cols)
            # plt.plot(dog1)
            plt.figure()
            plt.plot(rows)
            plt.plot(dog2)
            plt.show()

    if False:
        plt.figure()
        plt.plot(rotations, scores)
        plt.show()

    features['rotation'] = rotations[np.argmax(scores)]


def segment_module(im, features):
    h, w = im.shape
    cols = im[h // 10:-h // 10, :].mean(axis=0).astype(np.float32)
    cols /= cols.max()
    left, right = np.where(cols > 0.5)[0][[0, -1]]
    rows = im[:, w // 10:-w // 10].mean(axis=1).astype(np.float32)
    rows /= rows.max()
    top, bottom = np.where(rows > 0.5)[0][[0, -1]]

    if False:
        ImageViewer(im[::2, ::2])
        plt.figure()
        plt.plot(cols, label='cols')
        plt.vlines([left, right], 0, 1)
        plt.figure()
        plt.plot(rows, label='rows')
        plt.vlines([top, bottom], 0, 1)
        plt.legend()
        plt.show()

    # use autocorrelation of foreground columns to determine cell size (busbars can interfere with rows)
    ac_cols = stattools.acf(cols[left:right], nlags=len(cols) - 1, fft=False, unbiased=False)
    min_cell_size = 250
    max_cell_size = 800
    cell_width = min_cell_size + np.argmax(ac_cols[min_cell_size:max_cell_size])

    ac_rows = stattools.acf(rows[top:bottom], nlags=len(rows) - 1, fft=False, unbiased=False)
    min_cell_size = 250
    max_cell_size = 800
    cell_height = min_cell_size + np.argmax(ac_rows[min_cell_size:max_cell_size])

    if abs(cell_width - cell_height) > 20:
        print "WARNING: Assuming cell height is incorrect. Diff: %d vs %d" % (cell_width, cell_height)
        cell_height = pixel_ops.HillClimb1D(ac_rows.astype(np.float32), cell_width)

    if False:
        print cell_width, cell_height
        plt.figure()
        plt.plot(ac_cols)
        plt.vlines([cell_width], 0, 1)
        plt.figure()
        plt.plot(ac_rows)
        plt.vlines([cell_height], 0, 1)
        plt.show()

    # find module columns
    num_cols = int(round((right - left) / float(cell_width)))
    start = (left + (right - (num_cols * cell_width))) // 2
    divisions_cols = [start + (i * cell_width) for i in range(num_cols + 1)]

    # optimize to shift with min mean value for cols (assumes dark space between cells)
    max_shift = 20
    shifts = np.arange(-max_shift, max_shift + 1)
    mean_vals = []
    for s in shifts:
        sample_locations = divisions_cols + s
        sample_locations = sample_locations[((sample_locations > 0) & (sample_locations < w))]
        mean_vals.append(cols[sample_locations].mean())
    divisions_cols += shifts[np.argmin(mean_vals)]

    if False:
        print shifts[np.argmin(mean_vals)]
        plt.figure()
        plt.plot(shifts, mean_vals)
        plt.figure()
        plt.plot(cols, label='cols')
        plt.vlines(divisions_cols, 0, 1)
        plt.show()

    # find module rows
    num_rows = int(round((bottom - top) / float(cell_height)))
    start = (top + (bottom - (num_rows * cell_height))) // 2
    divisions_rows = [start + (i * cell_height) for i in range(num_rows + 1)]

    # optimize to shift with min mean value for rows (assumes dark space between cells)
    shifts = np.arange(-max_shift, max_shift + 1)
    mean_vals = []
    for s in shifts:
        sample_locations = divisions_rows + s
        sample_locations = sample_locations[((sample_locations > 0) & (sample_locations < h))]
        mean_vals.append(rows[sample_locations].mean())
    divisions_rows += shifts[np.argmin(mean_vals)]

    if False:
        print shifts[np.argmin(mean_vals)]
        plt.figure()
        plt.plot(shifts, mean_vals)
        plt.figure()
        plt.plot(rows, label='rows')
        plt.vlines(divisions_rows, 0, 1)
        plt.show()

    # for each row of cells, fine tune left/right
    module_cell_props = {}
    num_rows = len(divisions_rows) - 1
    num_cols = len(divisions_cols) - 1
    num_cells = num_rows * num_cols
    for c in range(num_cells):
        module_cell_props[c] = {'border_cell': False}

    for r in range(num_rows):
        top = divisions_rows[r]
        bottom = divisions_rows[r + 1]
        row_im = im[top:bottom, :]
        col_profile = ndimage.gaussian_filter1d(row_im.mean(axis=0), sigma=3)
        col_profile_inv = col_profile * -1

        # optimize location
        divisions_cols_op = divisions_cols[:]
        for i in range(1, num_cols):
            x = divisions_cols_op[i]
            new_x = pixel_ops.HillClimb1D(col_profile_inv, x)
            if abs(new_x - x) < 10:
                divisions_cols_op[i] = new_x

        for c in range(num_cols):
            cell_num = (r * num_cols) + c
            module_cell_props[cell_num]['crop_left'] = divisions_cols_op[c]
            module_cell_props[cell_num]['crop_right'] = divisions_cols_op[c + 1] + 1
            if c in [0, num_cols - 1]:
                module_cell_props[cell_num]['border_cell'] = True

        if False:
            plt.figure()
            plt.plot(col_profile)
            plt.vlines(divisions_cols, ymin=col_profile.min(), ymax=col_profile.max(), colors='r')
            plt.vlines(divisions_cols_op, ymin=col_profile.min(), ymax=col_profile.max(), colors='g')
            view = ImageViewer(row_im)
            view.show()

    # for each column of cells, fine tune top/bottom.
    for c in range(num_cols):
        left = divisions_cols[c]
        right = divisions_cols[c + 1]
        col_im = im[:, left:right]
        row_profile = ndimage.gaussian_filter1d(col_im.mean(axis=1), sigma=3)
        row_profile_inv = row_profile * -1

        # optimize location
        divisions_rows_op = divisions_rows[:]
        for i in range(1, num_rows):
            x = divisions_rows_op[i]
            new_x = pixel_ops.HillClimb1D(row_profile_inv, x)
            if abs(new_x - x) < 10:
                divisions_rows_op[i] = new_x

        for r in range(num_rows):
            cell_num = (r * num_cols) + c
            cell_name = "%s%d" % (string.ascii_uppercase[c], r + 1)
            module_cell_props[cell_num]['cell_name'] = cell_name
            module_cell_props[cell_num]['crop_top'] = divisions_rows_op[r]
            module_cell_props[cell_num]['crop_bottom'] = divisions_rows_op[r + 1] + 1
            if r in [0, num_rows - 1]:
                module_cell_props[cell_num]['border_cell'] = True

        if False:
            plt.figure()
            plt.plot(row_profile)
            plt.vlines(divisions_rows, ymin=row_profile.min(), ymax=row_profile.max(), colors='r')
            plt.vlines(divisions_rows_op, ymin=row_profile.min(), ymax=row_profile.max(), colors='g')
            view = ImageViewer(col_im)
            ImageViewer(col_im[divisions_rows_op[0]:divisions_rows_op[1], :])
            view.show()

    features['_divisions_rows'] = divisions_rows
    features['_divisions_cols'] = divisions_cols
    features['num_rows'] = num_rows
    features['num_cols'] = num_cols
    features['num_cells'] = num_cells
    features['_cell_properties'] = module_cell_props

    # get median cell width/height
    widths = []
    heights = []
    for bounds in module_cell_props.values():
        widths.append(bounds['crop_right'] - bounds['crop_left'])
        heights.append(bounds['crop_bottom'] - bounds['crop_top'])
    features['cell_width'] = int(round(np.median(widths)))
    features['cell_height'] = int(round(np.median(heights)))

    # get a tighter fit for the ones along the edges
    for (cell_num, cell_props) in module_cell_props.iteritems():
        if cell_num < num_cols:
            cell_props['crop_top'] = cell_props['crop_bottom'] - features['cell_height']
        elif cell_num > (num_rows - 1) * num_cols:
            cell_props['crop_bottom'] = cell_props['crop_top'] + features['cell_height']
        if cell_num % num_cols == 0:
            cell_props['crop_left'] = cell_props['crop_right'] - features['cell_width']
        elif cell_num % num_cols == num_cols - 1:
            cell_props['crop_right'] = cell_props['crop_left'] + features['cell_width']


def register_images(im_pl, im_el):
    # left-right
    def dip_strength(im, axis=0, diff=10):
        profile = im.mean(axis=axis)
        dip_profile = np.minimum(np.roll(profile, diff), np.roll(profile, -diff)) - profile
        dip_profile[[0, -1]] = 0
        dip_profile[dip_profile < 0] = 0
        dip_profile = ndimage.gaussian_filter1d(dip_profile, sigma=5)

        return profile, dip_profile

    cols_pl, ds_pl = dip_strength(im_pl, 0)
    cols_el, ds_el = dip_strength(im_el, 0)

    cor_vals = []
    shifts = range(-25, 26)
    for shift in shifts:
        cor_vals.append((ds_pl * np.roll(ds_el, shift)).sum())

    shift_h = shifts[np.argmax(cor_vals)]

    if False:
        print shift_h
        plt.figure()
        plt.plot(ds_pl)
        plt.plot(ds_el)
        plt.figure()
        plt.plot(shifts, cor_vals)
        plt.figure()
        plt.plot(cols_pl)
        plt.plot(np.roll(cols_el, shift_h))
        plt.show()

    return shift_h


def module_rotate(im, rotation=None):
    if rotation is None:
        # downsize
        max_side = max(im.shape)
        down_factor = max_side // 2500
        im_down = im[::down_factor, ::down_factor].astype(np.float32)

        # correct rotation
        features = {}
        find_module_rotation(im_down, features)
        rotation = features['rotation']

    if abs(rotation) > 0.01:
        h, w = im.shape
        dsize = (w, h)
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        im_rotated = np.empty((dsize[1], dsize[0]), np.float32)
        cv2.warpAffine(im, rot_mat, dsize, flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE, dst=im_rotated)

        if False:
            view = ImageViewer(im[::2, ::2])
            ImageViewer(im_rotated[::2, ::2])
            view.show()
    else:
        im_rotated = im.copy()

    return rotation, im_rotated


def profile_busbars(profile):
    # enhance busbar locations (for registration)
    w = len(profile) // 20
    profile[:w] = 0
    profile[-w:] = 0

    diff = len(profile) // 50
    dip_profile = np.minimum(np.roll(profile, diff), np.roll(profile, -diff)) - profile
    dip_profile[[0, -1]] = 0
    dip_profile[dip_profile < 0] = 0
    dip_profile = ndimage.gaussian_filter1d(dip_profile, sigma=3)

    return dip_profile


# def profile_fingers(profile):
#     # enhance finger locations (for registration)
#     w = len(profile) // 20
#     dip_profile = profile - ndimage.gaussian_filter1d(profile, sigma=10)
#     dip_profile[:w] = 0
#     dip_profile[-w:] = 0
#
#     if False:
#         plt.figure()
#         plt.plot(profile)
#         plt.plot(dip_profile)
#         plt.show()
#
#     return dip_profile


def register_to_template(cell, reference_cell_info, registration_props, debug=False):
    if 'col_profile' not in reference_cell_info:
        cell_ref = reference_cell_info['im']
        col_profile = profile_busbars(cell_ref.mean(axis=0))
        row_profile = cell_ref.mean(axis=1)
        row_profile /= row_profile.max()
        reference_cell_info['col_profile'] = col_profile
        reference_cell_info['row_profile'] = row_profile
        ref_profile_row_flat = row_profile.copy()
        ref_profile_row_flat[ref_profile_row_flat > 0.5] = 0.5
        reference_cell_info['row_profile_flat'] = ref_profile_row_flat

        # get period
        peaks = np.where(((row_profile > np.roll(row_profile, 1)) & (row_profile > np.roll(row_profile, -1))))[0]
        peaks = peaks[5:-5]
        period = int(round(np.median(peaks[1:] - peaks[:-1])))
        reference_cell_info['finger_period'] = period
        reference_cell_info['row_profile_smooth'] = ndimage.gaussian_filter1d(row_profile, sigma=period)
        reference_cell_info['row_profile_diff'] = row_profile - reference_cell_info['row_profile_smooth']

    ref_im = reference_cell_info['im']
    period = reference_cell_info['finger_period']

    # make the input and output the same size
    height_diff = cell.shape[0] - ref_im.shape[0]
    lines_top = abs(height_diff // 2)
    if height_diff > 0:
        cell = cell[lines_top:cell.shape[0] - height_diff + lines_top, :]
    elif height_diff < 0:
        cell = np.pad(cell, ((lines_top, (-1 * height_diff) - lines_top), (0, 0)), mode='edge')

    width_diff = cell.shape[1] - ref_im.shape[1]
    lines_left = abs(width_diff // 2)
    if width_diff > 0:
        cell = cell[:, lines_left:cell.shape[1] - width_diff + lines_left]
    elif width_diff < 0:
        cell = np.pad(cell, ((0, 0), (lines_left, (-1 * width_diff) - lines_left)), mode='edge')

    assert cell.shape == ref_im.shape

    if 'shift_h' not in registration_props:
        ref_profile_col = reference_cell_info['col_profile']
        ref_profile_row = reference_cell_info['row_profile']

        # left/right shift
        col_profile = profile_busbars(cell.mean(axis=0))
        s = len(col_profile) // 50
        cor_vals = []
        shifts = range(-s, s + 1)
        for shift in shifts:
            cor_vals.append((ref_profile_col * np.roll(col_profile, shift)).sum())

        # only pick if there is a local maximum
        if np.argmax(cor_vals) not in [0, len(cor_vals) - 1]:
            shift_h = shifts[np.argmax(cor_vals)]
        else:
            shift_h = 0

        if False:
            # measure distance between first and last busbar (to look for distortion)
            col_profile2 = col_profile / col_profile.max()
            peaks = np.where((col_profile2 > np.roll(col_profile2, 1)) &
                             (col_profile2 > np.roll(col_profile2, -1)) &
                             (col_profile2 > 0.5))[0]
            if len(peaks) != 4:
                plt.figure()
                plt.plot(col_profile2)
                plt.show()
            print peaks[-1] - peaks[0]

        if debug:
            print shift_h
            plt.figure()
            plt.plot(shifts, cor_vals)
            plt.figure()
            plt.plot(reference_cell_info['col_profile'], label="ref")
            plt.plot(col_profile, label="orig")
            plt.plot(np.roll(col_profile, shift_h), label="shifted")
            plt.legend()
            plt.show()

        # up/down shift
        # NOTE: This direction is harder since there aren't busbars to base alignment on.
        # 1. rough alignment of bright areas (anything > 0.5)
        # 2. local fine tune based on fingers
        row_profile = cell.mean(axis=1)
        row_profile /= row_profile.max()
        row_profile_flat = row_profile.copy()
        row_profile_flat[row_profile_flat > 0.5] = 0.5

        if debug:
            plt.figure()
            plt.plot(row_profile_flat)
            plt.plot(reference_cell_info['row_profile_flat'])
            plt.show()

        cor_vals = []
        for shift in shifts:
            cor_vals.append((reference_cell_info['row_profile_flat'] * np.roll(row_profile_flat, shift)).sum())

        # only pick if there is a local maximum
        if np.argmax(cor_vals) not in [0, len(cor_vals) - 1]:
            shift_v_rough = shifts[np.argmax(cor_vals)]
        else:
            shift_v_rough = 0

        if debug:
            print "Rough alignment"
            plt.figure()
            plt.plot(shifts, cor_vals)
            plt.show()

        row_profile_smooth = ndimage.gaussian_filter1d(row_profile, sigma=reference_cell_info['finger_period'])
        row_profile_diff = row_profile - row_profile_smooth

        cor_vals = []
        shifts = np.arange(shift_v_rough - (period // 2), shift_v_rough + (period // 2) + 1)
        # print shift_v_rough, shifts
        for shift in shifts:
            cor_vals.append((reference_cell_info['row_profile_diff'] * np.roll(row_profile_diff, shift)).sum())
        shift_v = shifts[np.argmax(cor_vals)]

        if debug:
            print shift_v
            plt.figure()
            plt.plot(ref_profile_row)
            plt.plot(row_profile)
            plt.figure()
            plt.plot(reference_cell_info['row_profile'], label='ref')
            plt.plot(row_profile, label='orig')
            plt.plot(np.roll(row_profile, shift_v), label='shifted')
            plt.legend()
            plt.figure()
            plt.plot(shifts, cor_vals)
            plt.show()
        registration_props['shift_h'] = shift_h
        registration_props['shift_v'] = shift_v
    else:
        # use provided params
        shift_h = registration_props['shift_h']
        shift_v = registration_props['shift_v']

    registered = ndimage.shift(cell, (shift_v, shift_h), mode='nearest')

    if False:
        print shift_v, shift_h
        # view = ImageViewer(cell)
        view = ImageViewer(ref_im)
        ImageViewer(registered)
        view.show()

    return registered


def cell_rotate(im, cell_crop_props, debug=False):
    # assume BBs are horizontal, and make vertical
    im = np.ascontiguousarray(np.rot90(im))
    rotated = np.empty_like(im)
    h, w = im.shape

    if 'crop_rotation' in cell_crop_props:
        rotation = cell_crop_props['crop_rotation']
        rotations = None
    else:
        rotations = np.linspace(-1, 1, 21)
        stds = []

        for r in rotations:
            rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), r, 1.0)
            cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, dst=rotated, borderValue=-1.0)
            if False:
                print r
                view = ImageViewer(rotated[30:-30, 30:-30])
                view.show()

            rows = rotated[30:-30, 30:-30].mean(axis=1)
            sigma = 4
            stds.append((rows - ndimage.gaussian_filter1d(rows, sigma=sigma)).std())
            if False:
                print r, stds[-1]
                plt.figure()
                plt.plot(rows)
                plt.plot(ndimage.gaussian_filter1d(rows, sigma=sigma))
                plt.show()

        rotation = rotations[np.argmax(stds)]

    if np.abs(rotation) > 0:

        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)
        cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE, borderValue=0.0, dst=rotated)
    else:
        rotated = im

    if debug:
        print rotation
        ImageViewer(im)
        ImageViewer(rotated)
        if rotations is not None:
            plt.figure()
            plt.plot(rotations, stds)
            plt.show()

    cell_crop_props['crop_rotation'] = rotation

    return rotated


def extract_cell_images(im_pl, im_el, features):
    # Use PL as this is more reliable for finding rotation
    t1 = timeit.default_timer()
    rotation_pl, im_pl_rotated = module_rotate(im_pl)
    t2 = timeit.default_timer()

    if False:
        print "  Rotation time: %0.02fs Angle: %0.02f" % (t2 - t1, rotation_pl)
        view = ImageViewer(im_pl[::2, ::2])
        ImageViewer(im_pl_rotated[::2, ::2])
        view.show()

    if im_el is not None:
        # Registration
        # - assume no between capture rotation. use same rotation as for PL image
        rotation_el, im_el_rotated = module_rotate(im_el, rotation_pl)

        if False:
            view = ImageViewer(im_pl_rotated[::4, ::4])
            ImageViewer(im_el_rotated[::4, ::4])
            view.show()

        # horizontal registration
        shift_h = register_images(im_pl_rotated, im_el_rotated)
        im_el_registered = np.roll(im_el_rotated, shift_h, axis=1)

        # vertical registration
        shift_v = register_images(im_pl_rotated.T, im_el_registered.T)
        im_el_registered = np.roll(im_el_registered, shift_v, axis=0)

        if False:
            print shift_v, shift_h
            view = ImageViewer(im_pl_rotated[::4, ::4])
            ImageViewer(im_el_registered[::4, ::4])
            view.show()
            sys.exit()

        # PL/EL ratio
        im_ratio = im_pl_rotated / np.maximum(0.01, im_el_registered)
        features['im_pl_el'] = im_ratio

        if False:
            view = ImageViewer(im_ratio[::4, ::4])
            view.show()
    else:
        im_ratio = None

    # segment individual cells
    segment_module(im_pl_rotated, features)
    if False:
        # not yet sure if there is any use for these
        features["_im_pl_cropped"] = im_pl_rotated[features['_divisions_rows'][0]:features['_divisions_rows'][-1],
                                                   features['_divisions_cols'][0]:features['_divisions_cols'][-1]]
        features["_im_el_cropped"] = im_el_rotated[features['_divisions_rows'][0]:features['_divisions_rows'][-1],
                                                   features['_divisions_cols'][0]:features['_divisions_cols'][-1]]
    features["_im_ratio_cropped"] = im_ratio[features['_divisions_rows'][0]:features['_divisions_rows'][-1],
                                             features['_divisions_cols'][0]:features['_divisions_cols'][-1]]

    # save cell images & create cell template
    # 1. need to pick a "good" one for reference template.
    #   - It should be interior.
    #   - If picked at random, could be an abnormal one. Therefore, ranks cells by some feature (e.g. std)
    #     and pick the one that is in the middle
    module_cell_props = features['_cell_properties']
    cell_character = []
    for (cell_num, cell_props) in module_cell_props.iteritems():
        if cell_props['border_cell']:
            continue

        top = cell_props['crop_top']
        bottom = cell_props['crop_bottom']
        left = cell_props['crop_left']
        right = cell_props['crop_right']
        cell_pl = im_pl_rotated[top:bottom, left:right]
        cell_character.append((cell_num, cell_pl.std()))

        if False:
            print cell_props['cell_name']
            view = ImageViewer(cell_pl)
            view.show()

    # - this is a measure of how uniform the cells appear. if high variation, there is probably something wrong
    #  with the module, and trying to register each cell individually won't work well
    # - suitable threshold seems to be around 0.3
    # TODO: can we do something more robust that works in these cases?
    stds = np.array([c[1] for c in cell_character])
    cell_cov = stds.std() / stds.mean()
    features['cell_variation'] = cell_cov

    cell_character.sort(key=lambda x: x[1])
    cell_ref_num = cell_character[len(cell_character) // 2][0]
    cell_props = module_cell_props[cell_ref_num]
    cell_ref = im_pl_rotated[cell_props['crop_top'] - 1:cell_props['crop_bottom'] + 1,
               cell_props['crop_left'] - 1:cell_props['crop_right'] + 1]
    cell_ref = cell_rotate(cell_ref, {})
    cell_ref_data = {'im': cell_ref}

    if False:
        view = ImageViewer(cell_ref)
        view.show()

    cell_records = list(module_cell_props.iteritems())

    # create cell template:
    # 1. register with "reference" cell
    # 2. median of mean to make robust
    #    - too memory/computationally expensive to take median of all cells
    #    - therefore, take mean of batches, and median of mean?
    #    - alternative: median of sub-sample (might have less blurring?)
    registered_cells = np.empty((len(cell_records), cell_ref.shape[0], cell_ref.shape[1]))
    for e, (cell_num, cell_props) in enumerate(cell_records):
        if False:
            print cell_props['cell_name'],
        top = cell_props['crop_top']
        bottom = cell_props['crop_bottom']
        left = cell_props['crop_left']
        right = cell_props['crop_right']
        cell_pl = im_pl_rotated[top:bottom, left:right]
        cell_crop_props = {}
        rotated = cell_rotate(cell_pl, cell_crop_props, debug=cell_props['cell_name'] == 'D666')
        registered = register_to_template(rotated, cell_ref_data, cell_crop_props,
                                          debug=cell_props['cell_name'] == 'F66')

        cell_props['im_cell_pl'] = registered

        # extract ratio image
        if im_ratio is not None:
            cell_el = im_el[top:bottom, left:right]
            cell_el_rotated = cell_rotate(cell_el, cell_crop_props)
            cell_el_reg = register_to_template(cell_el_rotated, cell_ref_data, cell_crop_props)
            cell_props['im_cell_el'] = cell_el_reg

            cell_ratio = im_ratio[top:bottom, left:right]
            cell_ratio_rotated = cell_rotate(cell_ratio, cell_crop_props)
            cell_ratio_reg = register_to_template(cell_ratio_rotated, cell_ref_data, cell_crop_props)
            cell_props['im_cell_ratio'] = cell_ratio_reg

        if False:
            print "Cell: %s" % cell_props['cell_name'], top
            view = ImageViewer(cell_pl)
            ImageViewer(rotated)
            ImageViewer(registered)
            view.show()

        registered_cells[e, :, :] = registered

        if False:
            folder = r"C:\Users\Neil\Desktop\cells"
            fn_out = os.path.join(folder, "cell_%s.png" % cell_props['cell_name'])
            im_out = (ip.scale_image(registered) * 255).astype(np.uint8)
            ip.save_image(fn_out, im_out)

    # cell_template /= features['num_cells']
    cell_template = np.median(registered_cells, axis=0)
    # cell_template = np.mean(registered_cells, axis=0)

    features['im_00_template_u16'] = np.round(cell_template).astype(np.uint16)

    if False:
        template = (ip.scale_image(cell_template) * 255).astype(np.uint8)
        ip.save_image(r"C:\Users\Neil\Desktop\cell comp\t1.png", template)

        view = ImageViewer(cell_template)
        cell_num = 12
        if False and 'cell_pl_im' in module_cell_props[cell_num]:
            ImageViewer(module_cell_props[cell_num]['cell_pl_im'])
            ImageViewer(module_cell_props[cell_num]['cell_ratio_im'])
            ImageViewer(module_cell_props[cell_num]['cell_pl_im'] / cell_template)
        view.show()
        sys.exit()

    return features


def analyse_points(im):
    # this was used for finding distances between rows/cols in test patterns

    # quantize into 17 rows
    import pandas as pd
    import scipy.cluster.vq as vq
    from mpl_toolkits.mplot3d import Axes3D

    fn = r"C:\Users\Neil\Dropbox (Personal)\BT\Data\modules\points.csv"
    df = pd.read_csv(fn)
    xs, ys = df['X'].values, df['Y'].values

    np.random.seed(666)
    rows, _ = vq.kmeans(ys, k_or_guess=9, thresh=1)
    rows.sort()
    mid_xs = []
    mid_ys = []
    mid_diff = []
    marker_cols = None
    for e, r in enumerate(rows):
        mask = ((ys > r - 40) & (ys < r + 40))
        if mask.sum() not in [6, 12]:
            print r, mask.sum()
            sys.exit()
        xs_row = xs[mask]
        ys_row = ys[mask]

        if e == 0:
            marker_cols = xs_row

        sort_args = np.argsort(xs_row)
        xs_row = xs_row[sort_args]
        ys_row = ys_row[sort_args]
        for i in range(0, len(xs_row), 3):
            y = ys_row[i:i + 3].mean()
            x1 = (xs_row[i] + xs_row[i + 1]) / 2.0
            x2 = (xs_row[i + 1] + xs_row[i + 2]) / 2.0
            mid_xs.append(x1)
            mid_ys.append(y)
            mid_diff.append(xs_row[i + 1] - xs_row[i])

            mid_xs.append(x2)
            mid_ys.append(y)
            mid_diff.append(xs_row[i + 2] - xs_row[i + 1])

    if False:
        im = im[::2, ::2]
        xs /= 2
        ys /= 2
        print len(xs)
        plt.figure()
        plt.imshow(im, cmap=plt.cm.gray)
        plt.plot(xs, ys, 'o')
        plt.show()
        sys.exit()

    mid_xs = np.array(mid_xs, np.float32)
    mid_ys = np.array(mid_ys, np.float32)
    mid_diff = np.array(mid_diff, np.float32)
    mid_diff_mean = mid_diff.mean()

    if False:
        mid_diff /= mid_diff_mean
        mid_diff -= 1
    else:
        mid_diff -= mid_diff_mean

    plt.figure()
    plt.plot(xs, ys, 'o')
    plt.plot(mid_xs, mid_ys, 'o')
    plt.gca().invert_yaxis()
    # plt.figure()
    # mean_diffs = []
    # for i in range(8):
    #     s = i*8
    #     plt.plot(mid_xs[s:s+8], mid_diff[s:s+8])
    #     mean_diffs.append(np.mean(mid_diff[s:s+8]))
    # plt.figure()
    # plt.plot(mean_diffs)

    xx, yy = np.meshgrid(np.linspace(xs.min(), xs.max(), num=10),
                         np.linspace(ys.min(), ys.max(), num=10))

    if False:
        point = np.array([0, 0, 0])
        normal = np.array([0, 0, 1])
        d = -point.dot(normal)
        zz = (-normal[0] * xx - normal[1] * yy - d) * 1. / normal[2]
    else:
        # zz = np.ones_like(xx)*mid_diff.mean()

        def error(params):
            a, b, c, d = params
            dists = (np.abs(a * mid_xs + b * mid_ys + c * mid_diff + d) /
                     np.sqrt(a * a + b * b + c * c))

            return np.abs(dists).mean()

        params_op = optimize.fmin(error, (0.1, 0.1, 0.1, 0))
        a, b, c, d = params_op
        print params_op, error(params_op)
        zz = -1 * (a * xx + b * yy + + d) / c

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zz -= mid_diff.mean()
    mid_diff -= mid_diff.mean()
    ax.scatter(mid_xs, mid_ys, mid_diff)
    # ax.plot_surface(xx, yy, zz, color='blue', alpha=0.5)

    if False:
        y1, y2 = im.shape[0] - 1000, im.shape[0]
        xx, yy = np.meshgrid(range(y1, y2),
                             range(im.shape[1]))
    else:
        xx, yy = np.meshgrid(range(im.shape[0]),
                             range(im.shape[1]))
        xx = xx[::2, ::2]
        yy = yy[::2, ::2]

    im_spacing = -1 * (a * xx + b * yy + + d) / c
    im_spacing = im_spacing.T[::-1, ::-1]
    view = ImageViewer(im_spacing)

    plt.show()


def cell_processing(cell_props):
    f = {'_alg_mode': 'multi cell'}
    cell_name = cell_props['cell_name']
    cell_im = cell_props['im_cell']
    if False:
        print cell_name
    try:
        multi.feature_extraction(cell_im, f, already_cropped=True)
    except cell.MissingBusbarsException:
        print "BBs not found"

    return f


def analyse_module(features):
    im = np.ascontiguousarray(features["_im_ratio_cropped"])
    h, w = im.shape
    # mask out rows and columns
    border = 20
    border_cols = features['_divisions_cols'] - features['_divisions_cols'][0]
    for c in border_cols:
        im[:, max(c-border, 0):min(c+border+1, w)] = 0
    border_rows = features['_divisions_rows'] - features['_divisions_rows'][0]
    for r in border_rows:
        im[max(r - border, 0):min(r + border + 1, h), :] = 0

    # scale so max is around
    scale = ((2**15) / im.max())
    im *= scale

    f = {}
    hist = ip.histogram_percentiles(im, f, skip_zero=True)
    hist = hist[:f['hist_percentile_99.9']]
    hist_norm = hist / hist.max()
    lower = np.where(hist_norm > 0.02)[0][0]
    upper = 2*f['hist_peak'] - lower
    high_vals = (hist[upper:].sum() / float(hist.sum()))
    features['module_bright_area_fraction'] = high_vals

    if False:
        print "%s: %0.01f%%" % (features['fn'], high_vals)
        ip.print_metrics(f)
        plt.figure()
        plt.xlabel("PL/EL ratio")
        plt.ylabel("Count")
        plt.title("Above threshold: %0.02f%%" % high_vals)
        xs = np.arange(len(hist)) / float(scale)
        plt.plot(xs, hist)
        plt.vlines([upper / float(scale)], ymin=0, ymax=hist.max())
        if False:
            plt.savefig(os.path.join(r"C:\Users\Neil\Desktop\M1\hist", features['fn']+'_1.png'))
            im = features["_im_ratio_cropped"]
            im[im > f['hist_percentile_99']] = f['hist_percentile_99']
            ip.save_image(os.path.join(r"C:\Users\Neil\Desktop\M1\hist", features['fn']+'_0.png'), im)
        else:
            plt.show()
            view = ImageViewer(im[::3, ::3])
            view.show()
        sys.exit()


def feature_extraction(im_pl, im_el, features):
    t_start = timeit.default_timer()

    # TODO: check if entire EL image is dark. ratio of means?

    # segment the module into individual cells and find cell template
    module_props = extract_cell_images(im_pl, im_el, features)

    if False:
        folder_out = r"C:\Users\Neil\Desktop\module_cracks"
        module_id = features['fn'].split('_')[0]

        # save cell images
        for cell_props in features['_cell_properties'].values():
            cell_name = cell_props['cell_name']
            if ((module_id == "CNY-098" and cell_name in ['B3', 'G3', 'H3', 'E4']) or
                    (module_id == "CNY-101" and cell_name in ['C1', 'J1', 'H2', 'I2', 'J2']) or
                    (module_id == "CNY-139" and cell_name in ['A3']) or
                    (module_id == "CNY-232" and cell_name in ['G6']) or
                    (module_id == "CNY-449" and cell_name in ['D1', 'F1', 'A2', 'I2', 'A3', 'D4', 'E4', 'G5']) or
                    (module_id == "REC-143" and cell_name in ['H2', 'H4', 'B6', 'D6']) or
                    (module_id == "STP-410" and cell_name in ['K3', 'E6', 'H2', 'H3'])):
                im_pl = (ip.scale_image(cell_props['im_cell_pl'])*255).astype(np.uint8)
                fn_im = os.path.join(folder_out, "%s_%s_pl.png" % (module_id, cell_name))
                ip.save_image(fn_im, im_pl)

                im_el = (ip.scale_image(cell_props['im_cell_el']) * 255).astype(np.uint8)
                fn_im = os.path.join(folder_out, "%s_%s_el.png" % (module_id, cell_name))
                ip.save_image(fn_im, im_el)

                im_ratio = (ip.scale_image(cell_props['im_cell_ratio']) * 255).astype(np.uint8)
                fn_im = os.path.join(folder_out, "%s_%s_ratio.png" % (module_id, cell_name))
                ip.save_image(fn_im, im_ratio)

                fn_raw = os.path.join(folder_out, "%s_%s.npz" % (module_id, cell_name))
                np.savez_compressed(fn_raw, im_pl=cell_props['im_cell_pl'],
                                    im_el=cell_props['im_cell_el'],
                                    im_ratio=cell_props['im_cell_ratio'])

                if False:
                    print module_id, cell_name
                    view = ImageViewer(cell_props['im_cell_pl'])
                    ImageViewer(cell_props['im_cell_el'])
                    ImageViewer(cell_props['im_cell_ratio'])
                    view.show()
                    sys.exit()

    # module-level metrics
    analyse_module(features)

    if False:
        # cell processing
        if False:
            # process each cell individually
            for cell_props in features['_cell_properties'].values():
                cell_processing(cell_props)
        else:
            # process cells in parallel
            pool = mp.Pool(processes=4)
            results = [pool.apply_async(cell_processing, args=(cp,)) for cp in features['_cell_properties'].values()]
            [p.get() for p in results]
            pool.close()

    # save results
    # for cell_props in features['_cell_properties'].values():
    #     cell_name = cell_props['cell_name']
    #     cell_im = np.round(cell_props['im_cell']).astype(np.uint16)
    #     features['im_%s_pl_u16' % cell_name] = cell_im

    # TODO: check for dark cells/rows
    # - example ratio images

    # undo rotation
    if False and parameters.ORIGINAL_ORIENTATION:
        for feature in features.keys():
            if ((feature.startswith('im_') or feature.startswith('ov_') or
                    feature.startswith('bl_') or feature.startswith('mk_')) and features[feature].ndim == 2):
                features[feature] = features[feature].T[:, ::-1]

    # compute runtime
    t_stop = timeit.default_timer()
    features['processing_runtime'] = t_stop - t_start


def main():
    pass


if __name__ == '__main__':
    main()
