import features_stripes
import numpy as np
import image_processing as ip
from image_processing import ImageViewer


def run_stripe():
    if True:
        mode = "mono"
        # crack
        fn = r"C:\Users\Neil\BT\Data\stripe\2017-09-07 Baccini 1 in 1\S0041_20170907.120013_Baccini 1 in 1 test_ID2_raw.tif"
        # corner
        #fn = r"C:\Users\Neil\BT\Data\stripe\2017-09-07 Baccini 1 in 1\S0041_20170907.113711_Baccini 1 in 1_ID5_raw.tif"
    else:
        mode = "multi"
        fn = r"C:\Users\Neil\BT\Data\stripe\2017-09-07 Baccini 1 in 1\S0041_20170907.121040_Baccini 1 in 1 test_ID8_raw.tif"

    im_pl = ip.open_image(fn).astype(np.float32)
    features = {"mode": mode}
    features_stripes.feature_extraction(im_pl, features)
    rgb = features_stripes.create_overlay(features)
    ip.print_metrics(features)
    print ip.list_images(features)
    view = ImageViewer(im_pl)
    ImageViewer(features['bl_cropped_u8'])
    ImageViewer(rgb)
    view.show()


def main():
    run_stripe()


if __name__ == '__main__':
    main()
