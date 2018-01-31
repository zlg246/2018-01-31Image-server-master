import features_module
import numpy as np
import image_processing as ip
import os
from image_processing import ImageViewer


def run_module():
    if False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\REC-144\REC-144_G00_LR0086_P35_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\REC-144\REC-144_G00_LR0086_CC7.80_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\REC-143\REC-143_G00_LR0086_P35_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\REC-143\REC-143_G00_LR0086_CC7.50_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\CNY-232\CNY-232_G00_LR0106_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\CNY-232\CNY-232_G00_LR0106_CC13.00_2x2_EL.tif"
    elif True:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\STP-410\STP-410_G00_LR0052_P53_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\STP-410\STP-410_G00_LR0045_CC5.50_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\WIN-555\WIN-555_LR0245_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\WIN-555\WIN-555_LR0160_CV43.00_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\APO-217\APO-217_G00_LR0089_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\APO-217\APO-217_G00_LR0089_CC13.00_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\CNY-098\CNY-098_G00_LR0090_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\CNY-098\CNY-098_G00_LR0090_CC10.80_2x2_EL.tif"
    elif False:
        fn_el = r"C:\Users\Neil\BT\Data\modules\CNY-101\CNY-101_G00_LR0090_CC10.80_2x2_EL.tif"
        fn_pl = r"C:\Users\Neil\BT\Data\modules\CNY-101\CNY-101_G00_LR0090_P93_2x2_OCPL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\CNY-139\CNY-139_G00_LR0106_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\CNY-139\CNY-139_G00_LR0106_CC13.00_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\CNY-232\CNY-232_G00_LR0106_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\CNY-232\CNY-232_G00_LR0106_CC13.00_2x2_EL.tif"
    elif False:
        fn_pl = r"C:\Users\Neil\BT\Data\modules\CNY-449\CNY-449_G00_LR0106_P93_2x2_OCPL.tif"
        fn_el = r"C:\Users\Neil\BT\Data\modules\CNY-449\CNY-449_G00_LR0106_CC13.00_2x2_EL.tif"

    im_pl = ip.open_image(fn_pl).astype(np.float32)
    im_el = ip.open_image(fn_el).astype(np.float32)
    features = {'fn': os.path.splitext(os.path.split(fn_pl)[1])[0]}
    features_module.feature_extraction(im_pl, im_el, features)
    ip.print_metrics(features)
    ratio = features['im_pl_el']
    view = ImageViewer(ratio[::4, ::4])
    view.show()


def main():
    run_module()


if __name__ == '__main__':
    main()
