import numpy as np
from image_processing import ImageViewer
import image_processing as ip
import cv2
import matplotlib.pylab as plt
import glob
import os
import timeit
from scipy import ndimage, optimize
import parameters
import cropping
import pixel_ops


def crop(im):
    im_orig = im
    im = cv2.medianBlur(im, ksize=5)

    # these images have bright spots, so reduce range by:
    # - clipping: bottom 20% and top 20% (variations in this range have no useful info)
    # - square root
    vals = np.sort(im.flat)
    p20 = vals[int(0.05 * vals.shape[0])]
    p80 = vals[int(0.8 * vals.shape[0])]

    im[im > p80] = p80
    im[im < p20] = p20
    im -= p20
    im /= (p80 - p20)
    im = np.sqrt(im)

    if False:
        view = ImageViewer(im_orig)
        ImageViewer(im)
        view.show()

    return cropping.crop_wafer_cz(im, check_foreground=False, outermost_peak=False, create_mask=True)


def feature_extraction(im, features):
    t_start = timeit.default_timer()

    # crop
    crop_props = crop(im)
    features['corners'] = crop_props['corners']
    #print crop_props.keys()
    #features['crop_top'] = crop_props['crop_top']
    # features['corner_tl_x'] = crop_props['corners'][0][1]
    # features['corner_tl_y'] = crop_props['corners'][0][0]
    # features['corner_tr_x'] = crop_props['corners'][1][1]
    # features['corner_tr_y'] = crop_props['corners'][1][0]
    # features['corner_br_x'] = crop_props['corners'][2][1]
    # features['corner_br_y'] = crop_props['corners'][2][0]
    # features['corner_bl_x'] = crop_props['corners'][3][1]
    # features['corner_bl_y'] = crop_props['corners'][3][0]
    features['wafer_radius'] = crop_props['radius']
    features['_wafer_middle_orig'] = crop_props['center']
    features['crop_rotation'] = crop_props['estimated_rotation']
    cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE_CZ,
                                        fix_chamfer=False)
    if not cropped.flags['C_CONTIGUOUS']:
        cropped = np.ascontiguousarray(cropped)

    if False:
        view = ImageViewer(im)
        ImageViewer(cropped)
        view.show()

    # histogram features
    h, w = cropped.shape
    ip.histogram_percentiles(cropped, features, h // 2, w // 2, features['wafer_radius'])

    # normalise image
    min_val = features['hist_percentile_01'] / float(features['hist_percentile_99'])
    norm_upper = features['hist_percentile_99']
    norm_lower = min(0.2, min_val)
    normed = ((cropped / norm_upper) - norm_lower) / (1 - norm_lower)

    # calculate distance from wafer rotation middle
    r, theta = np.empty_like(normed, np.float32), np.empty_like(normed, np.float32)
    pixel_ops.CenterDistance(r, theta, h // 2, w // 2)
    features['im_center_dist_im'] = r

    # create mask: 1=background
    wafer_mask = np.zeros_like(cropped, np.uint8)
    pixel_ops.ApplyThresholdGT_F32_U8(features['im_center_dist_im'], wafer_mask, features['wafer_radius'], 1)
    features['bl_cropped_u8'] = wafer_mask

    features['im_cropped_u8'] = (np.clip(normed, 0.0, 1.0) * 255).astype(np.uint8)
    if cropped.dtype.type is np.uint16:
        features['im_cropped_u16'] = cropped
    else:
        features['im_cropped_u16'] = cropped.astype(np.uint16)

    # compute runtime
    t_stop = timeit.default_timer()
    features['runtime'] = t_stop - t_start

    return crop_props


def main():
    folder = r"C:\Users\Neil\BT\Data\half processed"
    files = glob.glob(os.path.join(folder, "*.tif"))
    for e, fn in enumerate(files):
        #if e != 34:
        #    continue
        print "%s (%d/%d)" % (fn, e, len(files))
        features = {}
        im = ip.open_image(fn).astype(np.float32)
        crop_props = feature_extraction(im, features)

        if True:
            # save crop results
            pil_im = cropping.draw_crop_box(im, crop_props, pil_im=True)
            fn_root = os.path.splitext(os.path.split(fn)[1])[0]
            fn_out = os.path.join(r"C:\Users\Neil\Desktop\results\crop", fn_root + ".png")
            pil_im.save(fn_out)


if __name__ == '__main__':
    main()
