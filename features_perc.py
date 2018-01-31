import numpy as np
from image_processing import ImageViewer
import image_processing as ip
from scipy import ndimage
import sys
import parameters
import pixel_ops
import cell_processing as cell
import cropping
import timeit
import features_cz_wafer as cz_wafer
import features_cz_cell as cz_cell
import features_multi_cell as cell_multi
import cv2

DARK_CRACK = 1
DARK_SHUNT = 2
DARK_UNKNOWN = 4


def create_overlay(features):
    if 'im_cropped_u8' not in features:
        return None

    im_u8 = features['im_cropped_u8']
    im_rgb = np.empty((im_u8.shape[0], im_u8.shape[1], 3), np.float32)
    im_rgb[:, :, :] = im_u8[:, :, np.newaxis]

    # bright lines
    if "ov_lines_horizontal_u8" in features:
        horizontal = features["ov_lines_horizontal_u8"]
        im_rgb[:, :, 2] += horizontal
        im_rgb[:, :, 1] -= 0.5 * horizontal
        im_rgb[:, :, 0] -= 0.5 * horizontal

    if "ov_lines_vertical_u8" in features:
        vertical = features["ov_lines_vertical_u8"]
        im_rgb[:, :, 0] -= vertical
        im_rgb[:, :, 1] += 0.5 * vertical
        im_rgb[:, :, 2] -= 0.5 * vertical

    if "mk_cracks_u8" in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_cracks_u8'], 'r')

    if "mk_dark_spots_outline_u8" in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_dark_spots_outline_u8'], 'g')

    if 'ov_dark_middle_u8' in features:
        impure = features["ov_dark_middle_u8"] // 2
        im_rgb[:, :, 0] += impure
        im_rgb[:, :, 1] -= impure
        im_rgb[:, :, 2] += impure

    im_rgb[im_rgb < 0] = 0
    im_rgb[im_rgb > 255] = 255
    im_rgb = im_rgb.astype(np.uint8)

    return im_rgb


def dark_spot_props(win_orig, win_flat, mask_pixels, ys, xs, y, x, h, w):
    struct = ndimage.generate_binary_structure(2, 1)
    mask_crack2 = ndimage.binary_dilation(mask_pixels, struct, iterations=1)
    mask_crack3 = ndimage.binary_dilation(mask_crack2, struct, iterations=1)
    defect_outline = mask_crack3 - mask_crack2

    if False:
        view = ImageViewer(win_flat)
        ImageViewer(mask_pixels)
        ImageViewer(defect_outline)
        view.show()

    # compute some features of the defect that will be used for classification
    defect_features = {}
    defect_features['strength_median'] = np.median(win_orig[defect_outline]) - np.median(win_orig[mask_pixels])
    defect_features['strength_mean'] = win_orig[defect_outline].mean() - win_orig[mask_pixels].mean()
    defect_features['strength_median_flat'] = np.median(win_flat[defect_outline]) - np.median(win_flat[mask_pixels])
    defect_features['strength_mean_flat'] = win_flat[defect_outline].mean() - win_flat[mask_pixels].mean()
    defect_features['strength_flat_max'] = win_flat.min() * -1
    defect_features['num_pixels'] = mask_pixels.sum()
    defect_features['edge_dist'] = min(x, y, w - 1 - x, h - 1 - y)
    defect_features['aspect_ratio'] = (max(ys.max() - ys.min(), xs.max() - xs.min()) /
                                       float(max(1, min(ys.max() - ys.min(), xs.max() - xs.min()))))
    defect_features['fill_ratio'] = defect_features['num_pixels'] / float(mask_pixels.shape[0] * mask_pixels.shape[1])
    defect_features['location_y'] = y
    defect_features['location_x'] = x

    return defect_features


def dark_spots(features):
    im = features['im_no_fingers']
    h, w = im.shape

    im_mini = im[::6, ::6]
    im_mini_med = cv2.medianBlur(im_mini, ksize=5)
    im_mini_smooth = cv2.GaussianBlur(im_mini_med, ksize=(0, 0), sigmaX=1)
    background = cv2.resize(im_mini_smooth, (w, h))
    dark_areas = background - im
    pixel_ops.ApplyThresholdLT_F32(dark_areas, dark_areas, 0.0, 0.0)

    foreground_mask = ((features['bl_cropped_u8'] == 0) | (features['bl_cropped_u8'] == 4))
    structure = ndimage.generate_binary_structure(2, 1)
    foreground_mask = ndimage.binary_erosion(foreground_mask, structure=structure, iterations=3)
    dark_areas[~foreground_mask] = 0

    DARK_SPOT_SENSITIVITY = 0.08
    dark_spots = (dark_areas > DARK_SPOT_SENSITIVITY).astype(np.uint8)
    min_size = int(h * w * 0.0001)
    ip.remove_small_ccs(dark_spots, min_size)

    dark_spots_outline = ndimage.binary_dilation(dark_spots, structure=structure, iterations=2) - dark_spots
    features['mk_dark_spots_filled_u8'] = dark_spots
    features['mk_dark_spots_outline_u8'] = dark_spots_outline

    if False:
        view = ImageViewer(im)
        ImageViewer(background)
        ImageViewer(dark_spots)
        ImageViewer(ip.overlay_mask(im, dark_spots_outline))
        view.show()


def bright_lines(features):
    im = features['im_no_fingers']
    h, w = im.shape
    if 'finger_period_row' in features:
        rh = int(round(features['finger_period_row']))
        cw = int(round(features['finger_period_col']))
    else:
        rh = int(round(features['finger_period']))
        cw = int(round(features['finger_period']))

    f_v = im - np.maximum(np.roll(im, shift=2 * cw, axis=1),
                          np.roll(im, shift=-2 * cw, axis=1))
    pixel_ops.ApplyThresholdLT_F32(f_v, f_v, 0.0, 0.0)

    # filter
    mask = (f_v > 0.02).astype(np.uint8)
    min_size = 0.0005 * h * w
    ip.remove_small_ccs(mask, min_size)
    f_v[mask == 0] = 0
    # features['_f_v'] = f_v.copy()

    f_h = im - np.maximum(np.roll(im, shift=2 * rh, axis=0),
                          np.roll(im, shift=-2 * rh, axis=0))
    pixel_ops.ApplyThresholdLT_F32(f_h, f_h, 0.0, 0.0)

    # filter
    mask = (f_h > 0.02).astype(np.uint8)
    min_size = 0.0005 * h * w
    ip.remove_small_ccs(mask, min_size)
    f_h[mask == 0] = 0
    # features['_f_h'] = f_h.copy()

    # normalize
    f_h /= 0.3
    f_v /= 0.3

    pixel_ops.ClipImage(f_h, 0.0, 1.0)
    pixel_ops.ClipImage(f_v, 0.0, 1.0)
    features['ov_lines_horizontal_u8'] = (f_h * 255).astype(np.uint8)
    features['ov_lines_vertical_u8'] = (f_v * 255).astype(np.uint8)

    features['bright_lines_horizontal'] = f_h.mean() * 100
    features['bright_lines_vertical'] = f_v.mean() * 100

    if False:
        view = ImageViewer(im)
        ImageViewer(f_v)
        ImageViewer(f_h)
        view.show()
        sys.exit()


def wafer_features(features):
    gridless = features['im_no_figners_bbs']
    cz_wafer.rds(gridless, features)
    cz_wafer.process_rings(gridless, features)
    cz_wafer.radial_profile(gridless, features)
    cz_wafer.dark_middle(gridless, features)


def feature_extraction(im, features, already_cropped=False):
    t_start = timeit.default_timer()

    if already_cropped:
        # features['crop_rotation'] = 0
        # features['cell_rotated'] = False
        rotated = cropping.correct_cell_rotation(im, features, already_cropped=already_cropped)
        cropped = cropping.crop_cell(rotated, im, features, width=None, already_cropped=True)
    else:
        # rotation & cropping
        rotated = cropping.correct_cell_rotation(im, features)
        cropped = cropping.crop_cell(rotated, im, features, width=None)

    features['im_cropped_u16'] = cropped.astype(np.uint16)
    h, w = cropped.shape

    if False:
        view = ImageViewer(im)
        ImageViewer(cropped)
        ImageViewer(features['im_cropped_u16'])
        view.show()
        sys.exit()

    # determine properties of the cell pattern
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

    if 'input_param_skip_features' not in features or int(features['input_param_skip_features']) != 1:
        wafer_features(features)
        bright_lines(features)
        if False:
            dark_spots(features)
        else:
            cz_cell.dark_spots(features)
        if True:
            temp = parameters.CELL_CRACK_STRENGTH
            parameters.CELL_CRACK_STRENGTH = 3.0
            cell.mono_cracks(features)
            parameters.CELL_CRACK_STRENGTH = temp

    # undo rotation
    if parameters.ORIGINAL_ORIENTATION and features['cell_rotated']:
        for feature in features.keys():
            if ((feature.startswith('im_') or feature.startswith('mask_') or
                     feature.startswith('map_') or feature.startswith('ov_') or
                     feature.startswith('bl_') or feature.startswith('mk_'))
                    and features[feature].ndim == 2):
                features[feature] = features[feature].T[:, ::-1]

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start


def create_overlay_multi(features):
    return cell_multi.create_overlay(features)


def feature_extraction_multi(im, features, already_cropped=False):
    cell_multi.feature_extraction(im, features, already_cropped=already_cropped)


def main():
    pass


if __name__ == '__main__':
    main()
