import glob
import os
import image_processing as ip
import cropping
import numpy as np
from image_processing import ImageViewer
from pprint import pprint
import parameters


def run_cropping(files, mode=None, display=True):
    for e, fn in enumerate(files):
        print "%s (%d/%d)" % (fn, e, len(files))
        features = {}
        im = ip.open_image(fn).astype(np.float32)
        if mode == "cell":
            rotated = cropping.correct_cell_rotation(im, features, already_cropped=False)
            cropped = cropping.crop_cell(rotated, im, features, width=None, already_cropped=False)
        elif mode == "mono wafer":
            features['_alg_mode'] = 'mono wafer'
            crop_props = cropping.crop_wafer_cz(im, create_mask=True, skip_crop=False)
            features.update(crop_props)

            cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE_CZ,
                                                fix_chamfer=False)

        if False:
            # save crop results
            pil_im = cropping.draw_crop_box(im, features, mode="pil")
            fn_root = os.path.splitext(os.path.split(fn)[1])[0]
            fn_out = os.path.join(r"C:\Users\Neil\Desktop\results\crop", fn_root + ".png")
            pil_im.save(fn_out)
        else:
            rgb = cropping.draw_crop_box(im, features, mode="rgb")
            pprint(features)
            view = ImageViewer(rgb)
            view.show()


def main():
    if False:
        files = [r"C:\Users\Neil\BT\Data\C3\perc\mono\BAC_1024_100\20150910_122155.612_BAC_1024_100_201.tif"]
        mode = "cell"
    else:
        mode = "mono wafer"
        files = [r"C:\Users\Neil\Desktop\outlines\mode84.tif"]

    run_cropping(files, mode=mode, display=True)


if __name__ == '__main__':
    main()
