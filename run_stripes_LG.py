import image_processing as ip
import numpy as np
from image_processing import ImageViewer
import glob
import os
import pandas as pd
import features_stripes


def run_single(fn, display=True, downsize=True):

    if False:
        mode = "mono"
    else:
        mode = "multi"

    features = {"_cell_type": mode}
    im = ip.open_image(fn).astype(np.float32)

    if False:
        view = ImageViewer(im)
        view.show()

    skip_crop = True
    features_stripes.feature_extraction(im, features, skip_crop)
    f = ip.print_metrics(features)
    if display:
        view = ImageViewer(im)
        rgb = features_stripes.create_overlay(features)
        ImageViewer(rgb)
        view.show()

    return f


def run_folder(folder):
    files = glob.glob(os.path.join(folder, "*.tif"))
    results = []
    for fn in files:
        print(fn)
        results.append(run_single(fn, display=False))

    df = pd.DataFrame(results)
    df.to_csv(r"crop_stripe/results.csv")


def main():
    if True:
        fn = r"crop_stripe/t4_8.tif"
        run_single(fn)
    else:
        folder = r"crop_stripe"
        run_folder(folder)


if __name__ == '__main__':
    main()