import numpy as np
from image_processing import ImageViewer
import image_processing as ip
import cell_processing as cell
import timeit
import cropping
import matplotlib.pylab as plt
import pixel_ops
from scipy import stats
import features_multi_cell as multi_cell
import parameters


def create_overlay(features):
    im_u8 = features['im_cropped_u8']
    im_rgb = np.empty((im_u8.shape[0], im_u8.shape[1], 3), np.float32)
    im_rgb[:, :, :] = im_u8[:, :, np.newaxis]

    if 'mk_cracks_u8' in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_cracks_u8'], 'r')

    im_rgb[im_rgb < 0] = 0
    im_rgb[im_rgb > 255] = 255
    im_rgb = im_rgb.astype(np.uint8)

    return im_rgb


def feature_extraction(im, features, already_cropped=False):
    t_start = timeit.default_timer()

    if 'input_param_num_stripes' in features:
        num_stripes = features['input_param_num_stripes']
    else:
        num_stripes = 6

    features['num_rows'] = 1
    features['num_cols'] = num_stripes

    if 'input_param_multi' in features:
        multi = int(features['input_param_multi']) == 1
    else:
        multi = False

    # rotation & cropping
    rotated = cropping.correct_cell_rotation(im, features, already_cropped=already_cropped)
    cropped = cropping.crop_cell(rotated, im, features, width=None, already_cropped=already_cropped)

    # stripe corners
    corner_tr_x = features['corner_tr_x']
    corner_tr_y = features['corner_tr_y']
    corner_tl_x = features['corner_tl_x']
    corner_tl_y = features['corner_tl_y']
    corner_br_x = features['corner_br_x']
    corner_br_y = features['corner_br_y']
    corner_bl_x = features['corner_bl_x']
    corner_bl_y = features['corner_bl_y']
    if features['cell_rotated']:
        x_diff_l = corner_bl_x - corner_tl_x
        y_diff_l = corner_bl_y - corner_tl_y
        x_diff_r = corner_br_x - corner_tr_x
        y_diff_r = corner_br_y - corner_tr_y
        for i in range(num_stripes):
            p_start = i / float(num_stripes)
            p_stop = (i + 1) / float(num_stripes)
            features["%02d_corner_tl_y" % (i + 1)] = int(round(corner_tl_y + (p_start * y_diff_l)))
            features["%02d_corner_tl_x" % (i + 1)] = int(round(corner_tl_x + (p_start * x_diff_l)))
            features["%02d_corner_bl_y" % (i + 1)] = int(round(corner_tl_y + (p_stop * y_diff_l)))
            features["%02d_corner_bl_x" % (i + 1)] = int(round(corner_tl_x + (p_stop * x_diff_l)))

            features["%02d_corner_tr_y" % (i + 1)] = int(round(corner_tr_y + (p_start * y_diff_r)))
            features["%02d_corner_tr_x" % (i + 1)] = int(round(corner_tr_x + (p_start * x_diff_r)))
            features["%02d_corner_br_y" % (i + 1)] = int(round(corner_tr_y + (p_stop * y_diff_r)))
            features["%02d_corner_br_x" % (i + 1)] = int(round(corner_tr_x + (p_stop * x_diff_r)))
    else:
        x_diff_t = corner_tr_x - corner_tl_x
        y_diff_t = corner_tr_y - corner_tl_y
        x_diff_b = corner_br_x - corner_bl_x
        y_diff_b = corner_br_y - corner_bl_y
        for i in range(num_stripes):
            p_start = i / float(num_stripes)
            p_stop = (i + 1) / float(num_stripes)
            features["%02d_corner_tl_y" % (i + 1)] = int(round(corner_tl_y + (p_start * y_diff_t)))
            features["%02d_corner_tl_x" % (i + 1)] = int(round(corner_tl_x + (p_start * x_diff_t)))
            features["%02d_corner_tr_y" % (i + 1)] = int(round(corner_tl_y + (p_stop * y_diff_t)))
            features["%02d_corner_tr_x" % (i + 1)] = int(round(corner_tl_x + (p_stop * x_diff_t)))

            features["%02d_corner_bl_y" % (i + 1)] = int(round(corner_bl_y + (p_start * y_diff_b)))
            features["%02d_corner_bl_x" % (i + 1)] = int(round(corner_bl_x + (p_start * x_diff_b)))
            features["%02d_corner_br_y" % (i + 1)] = int(round(corner_bl_y + (p_stop * y_diff_b)))
            features["%02d_corner_br_x" % (i + 1)] = int(round(corner_bl_x + (p_stop * x_diff_b)))

    features['im_cropped_u16'] = cropped.astype(np.uint16)
    h, w = cropped.shape

    corner_mask = np.ones_like(cropped, np.uint8)
    r, theta = np.empty_like(cropped, np.float32), np.empty_like(cropped, np.float32)
    pixel_ops.CenterDistance(r, theta, features['wafer_middle_y'], features['wafer_middle_x'])
    pixel_ops.ApplyThresholdGT_F32_U8(r, corner_mask, features['wafer_radius'], 0)

    if False:
        print features['cell_rotated']
        plt.figure()
        plt.plot(cropped.mean(axis=0))
        view = ImageViewer(im)
        ImageViewer(rotated)
        ImageViewer(cropped)
        ImageViewer(corner_mask)
        view.show()

    ip.histogram_percentiles(cropped, features, center_y=h // 2, center_x=w // 2, radius=features['wafer_radius'])
    cell.normalise(cropped, features)

    # find cell structure
    f = features.copy()
    cell.cell_structure(cropped, f)
    features['bl_cropped_u8'] = f['bl_cropped_u8']
    ip.histogram_percentiles(cropped, f, center_y=h // 2, center_x=w // 2, radius=features['wafer_radius'])
    cell.normalise(cropped, f)
    cell.remove_cell_template(f['im_norm'], f)

    if 'input_param_skip_features' not in features or int(features['input_param_skip_features']) != 1:
        if multi:
            # efficiency analysis
            multi_cell.bright_areas(f)
            multi_cell.efficiency_analysis(f)

            # save results
            features['impure_area_fraction'] = f['impure_area_fraction']
            features['dislocation_area_fraction'] = f['dislocation_area_fraction']

            im_dislocations = f['_foreground']
            dislocation_thresh = f['_dislocation_thresh']
            im_impure = f['_impure']
            impure_thresh = f['_impure_thresh']
        else:
            # cracks
            cell.mono_cracks(f)
            features['mk_cracks_u8'] = f['mk_cracks_u8']
            features['defect_count'] = f['defect_count']
            features['defect_present'] = f['defect_present']
            features['defect_length'] = f['defect_length']
            crack_skel = f['_crack_skel']

        # extract stats from each stripe
        stripe_width = w // num_stripes
        for s in range(num_stripes):
            s1 = int(round(s * stripe_width))
            s2 = int(round(min(w, (s + 1) * stripe_width)))
            stripe = cropped[:, s1:s2]
            mask = corner_mask[:, s1:s2]

            vals = stripe[mask == 1].flat

            features["%02d_hist_harmonic_mean" % (s + 1)] = 1.0 / (1.0 / np.maximum(0.01, vals)).mean()
            features["%02d_hist_median" % (s + 1)] = np.median(vals)
            features["%02d_hist_mean" % (s + 1)] = np.mean(vals)
            features["%02d_hist_percentile_01" % (s + 1)] = stats.scoreatpercentile(vals, 1)
            features["%02d_hist_percentile_99" % (s + 1)] = stats.scoreatpercentile(vals, 99)
            features["%02d_hist_std" % (s + 1)] = np.std(vals)
            features["%02d_hist_cov" % (s + 1)] = features["%02d_hist_std" % (s + 1)] / max(1, features[
                "%02d_hist_mean" % (s + 1)])

            if 'input_param_no_stripe_images' not in features or int(features['input_param_no_stripe_images']) != 1:
                features['im_%02d_u16' % (s + 1)] = stripe.astype(np.uint16)
                features['bl_%02d_cropped_u8' % (s + 1)] = features['bl_cropped_u8'][:, s1:s2]

            if False:
                view = ImageViewer(stripe)
                ImageViewer(mask)
                view.show()

            if multi:
                impure_stripe = im_impure[:, s1:s2]
                dis_stripe = im_dislocations[:, s1:s2]
                features["%02d_dislocation" % (s + 1)] = (dis_stripe > dislocation_thresh).mean()
                features["%02d_impure" % (s + 1)] = (impure_stripe < impure_thresh).mean()
            else:
                crack_stripe = features['mk_cracks_u8'][:, s1:s2]

                if 'input_param_no_stripe_images' not in features or int(features['input_param_no_stripe_images']) != 1:
                    features['mk_%02d_cracks_u8' % (s + 1)] = np.ascontiguousarray(crack_stripe, dtype=np.uint8)

                skel_stripe = crack_skel[:, s1:s2]
                features["%02d_defect_length" % (s + 1)] = skel_stripe.sum()
                if features["%02d_defect_length" % (s + 1)] > 0:
                    _, num_ccs = ip.connected_components(crack_stripe)
                    features["%02d_defect_count" % (s + 1)] = num_ccs
                    features["%02d_defect_present" % (s + 1)] = 1 if num_ccs > 0 else 0
                else:
                    features["%02d_defect_count" % (s + 1)] = 0
                    features["%02d_defect_present" % (s + 1)] = 0

    # undo rotation
    if parameters.ORIGINAL_ORIENTATION and features['cell_rotated']:
        features['num_rows'], features['num_cols'] = features['num_cols'], features['num_rows']
        for feature in features.keys():
            if ((feature.startswith('im_') or feature.startswith('ov_') or
                    feature.startswith('bl_') or feature.startswith('mk_')) and features[feature].ndim == 2):
                features[feature] = features[feature].T[:, ::-1]

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start


def main():
    pass


if __name__ == '__main__':
    main()
