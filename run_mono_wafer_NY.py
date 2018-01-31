import image_processing as ip
import features_cz_wafer as mono_wafer
import numpy as np
import cropping
import parameters
from image_processing import ImageViewer
from scipy import ndimage


def main():
    features = {}
    fn = r"C:\Users\Neil\Desktop\R3 crack\raw PL images\cracked wafer PL image.tif"
    im = ip.open_image(fn).astype(np.float32)

    if im.shape[0] > 700:
        print '    WARNING: Image resized'
        im_max = im.max()
        im = ndimage.zoom(im, 0.5)
        if im.max() > im_max:
            im[im > im_max] = im_max

    if False:
        view = ImageViewer(im)
        view.show()

    features['_alg_mode'] = 'mono wafer'
    crop_props = cropping.crop_wafer_cz(im, create_mask=True, skip_crop=False)
    features['corners'] = crop_props['corners']
    cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE_CZ,
                                        fix_chamfer=False)
    mono_wafer.feature_extraction(cropped, crop_props, features=features)

    ip.print_metrics(features)
    rgb = mono_wafer.create_overlay(features)
    view = ImageViewer(rgb)
    view.show()


if __name__ == '__main__':
    main()
