import image_processing as ip
import features_multi_wafer as multi_wafer
import numpy as np
import cropping
import parameters
from image_processing import ImageViewer
import glob
import os
from scipy import ndimage
import pandas as pd


def run_single(fn, display=True, downsize=True):
    features = {}
    im = ip.open_image(fn).astype(np.float32)

    if downsize and im.shape[0] > 750:
        print '    WARNING: Image resized'
        im_max = im.max()
        im = ndimage.zoom(im, 0.5)
        if im.max() > im_max:
            im[im > im_max] = im_max

    if False:
        view = ImageViewer(im)
        view.show()

    parameters.SLOPE_MULTI_WAFER = True
    parameters.BORDER_ERODE = 3
    parameters.MIN_IMPURE_AREA = 0.01

    features['_alg_mode'] = 'multi wafer'
    features['_fn'] = os.path.splitext(os.path.split(fn)[1])[0]
    crop_props = cropping.crop_wafer(im, create_mask=True)
    features['corners'] = crop_props['corners']
    cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE)
    multi_wafer.feature_extraction(cropped, crop_props, features=features)
    multi_wafer.combined_features(features)
    rgb = multi_wafer.create_overlay(features)

    f = ip.print_metrics(features, display=display)
    if display:
        print "Wafer type: %s" % multi_wafer.WaferType.types[features['wafer_type']]
        view = ImageViewer(rgb)
        ImageViewer(im)
        view.show()

    return f, features['im_cropped_u8'], rgb


def run_folder(folder):
    files = glob.glob(os.path.join(folder, "*.tif"))
    folder_out = r"C:\Users\Neil\Desktop\results"
    for wafer_type in 'fully_impure', 'transition', 'middle', 'edge', 'corner':
        folder = os.path.join(folder_out, wafer_type)
        if not os.path.isdir(folder):
            os.makedirs(folder)

    parameters.SLOPE_MULTI_WAFER = True
    parameters.BORDER_ERODE = 3
    parameters.MIN_IMPURE_AREA = 0.01
    results = []
    for e, fn in enumerate(files):
        print "%s (%d/%d)" % (fn, e+1, len(files))
        f, cropped, rgb = run_single(fn, display=False, downsize=False)
        results.append(f)

        wafer_type = multi_wafer.WaferType.types[f['wafer_type']]
        folder = os.path.join(folder_out, wafer_type)
        fn_root = os.path.splitext(os.path.split(fn)[1])[0]
        fn_out = os.path.join(folder, fn_root + '_0.png')
        ip.save_image(fn_out, cropped)
        fn_out = os.path.join(folder, fn_root + '_2.png')
        ip.save_image(fn_out, rgb)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(folder_out, "results.csv"))


def main():
    if True:
        fn = r"C:\Users\Neil\BT\Data\multi wafer\DW Multi 2\20171218T142800_IDhigh impurity.tif"
        run_single(fn)
    else:
        #folder = r"C:\Users\Neil\Desktop\Questions_on_algo\for neil"
        folder = r"C:\Users\Neil\BT\Data\multi wafer\DW Multi 2"
        run_folder(folder)


if __name__ == '__main__':
    main()
