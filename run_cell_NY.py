import image_processing as ip
import features_multi_cell as multi_cell
import features_cz_cell as mono_cell
import numpy as np
import cropping
import parameters
from image_processing import ImageViewer
import glob
import os
from scipy import ndimage
import pandas as pd


def run_single(fn, mode, display=True, downsize=True):
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

    features['_fn'] = os.path.splitext(os.path.split(fn)[1])[0]

    if mode == "multi":
        features['_alg_mode'] = 'multi wafer'
        multi_cell.feature_extraction(im, features=features)
    elif mode == "mono":
        features['_alg_mode'] = 'mono wafer'
        mono_cell.feature_extraction(im, features=features)

    f = ip.print_metrics(features)
    if display:
        rgb = multi_cell.create_overlay(features)
        view = ImageViewer(im)
        ImageViewer(rgb)
        view.show()

    return f


def run_folder(folder):
    files = glob.glob(os.path.join(folder, "*.tif"))
    results = []
    for fn in files:
        print fn
        results.append(run_single(fn, display=False))

    df = pd.DataFrame(results)
    df.to_csv(r"C:\Users\Neil\Desktop\results.csv")


def main():
    if True:
        # run single
        if True:
            # multi
            mode = "multi"

            # broken fingers
            #fn = r"C:\Users\Neil\BT\Data\C3\multi\baccini\20151002_005214.038_1_1144337220.tif"
            fn = r"C:\Users\Neil\BT\Data\C3\multi\Jinko_1024_100\20150910_110012.126_Jinko_1024_100_112.tif"
            #fn = r"C:\Users\Neil\BT\Data\C3\multi\mystery\S0072_20151105.165041_!SPAN LOWPL_ID198_GRADEB3_BIN12.tif"
            #fn = r"C:\Users\Neil\BT\Data\C3\multi\AMAT_mc-Si_1024_100\20150909_200645.556_AMAT_mc-Si_1024_100_223.tif"
        else:
            # mono
            mode = "mono"

            # no busbars
            parameters.CELL_NO_BBS = True
            fn = r"C:\Users\Neil\BT\Data\C3\mono\no bbs\maintenanceBTI 25.tif"

        run_single(fn, mode=mode, display=True)
    else:
        #folder = r"C:\Users\Neil\Desktop\Questions_on_algo\for neil"
        folder = r"C:\Users\Neil\Desktop\tiff for all tests"
        run_folder(folder)


if __name__ == '__main__':
    main()
