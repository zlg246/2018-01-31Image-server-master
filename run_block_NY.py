import features_block
import numpy as np
import image_processing as ip
from scipy import ndimage, interpolate
from image_processing import ImageViewer
import matplotlib.pylab as plt


def run_plir():
    fn = r"C:\Users\Neil\BT\Data\2017-09-06 TransferFunctions.TXT"
    vals = features_block.load_transfer(fn)
    spline_plir, spline_nf, spline_sp, spline_lp = features_block.interpolate_transfer(vals, debug=False)

    if False:
        fn_sp = r"C:\Users\Neil\BT\Data\blocks\PLIR\Trina\2016-05-12\5.4V W (Uncalibrated PL Image) west short pass.tif"
        fn_lp = r"C:\Users\Neil\BT\Data\blocks\PLIR\Trina\2016-05-12\5.4V W (Uncalibrated PL Image) west long pass.tif"
        fn_nf = r"C:\Users\Neil\BT\Data\blocks\PLIR\Trina\2016-05-12\5.4V W (Uncalibrated PL Image) west no filter.tif"
    elif False:
        fn_sp = r"C:\Users\Neil\BT\Data\blocks\PLIR\marker\S0069_20170807.033044_ID4624_plg.meas.block.b3BL.north.sp.img.tif"
        fn_lp = r"C:\Users\Neil\BT\Data\blocks\PLIR\marker\S0069_20170807.033044_ID4624_plg.meas.block.b3BL.north.lp.img.tif"
        fn_nf = r"C:\Users\Neil\BT\Data\blocks\PLIR\marker\S0069_20170807.033044_ID4624_plg.meas.block.b3BL.north.std.img.tif"
    else:
        fn_sp = r"C:\Users\Neil\Desktop\Rietech.2.1172\tifs\plg.meas.block.b3bl.north.sp.img.tif"
        fn_lp = r"C:\Users\Neil\Desktop\Rietech.2.1172\tifs\plg.meas.block.b3bl.north.lp.img.tif"
        fn_nf = r"C:\Users\Neil\Desktop\Rietech.2.1172\tifs\plg.meas.block.b3pl.img.tif"

    im_sp = ip.open_image(fn_sp, cast_long=False).astype(np.float32)
    im_lp = ip.open_image(fn_lp, cast_long=False).astype(np.float32)
    im_pl = ip.open_image(fn_nf, cast_long=False).astype(np.float32)

    if False:
        im_sp = ndimage.zoom(im_sp, zoom=0.5)
        im_lp = ndimage.zoom(im_lp, zoom=0.5)

    features = {}
    features_block.plir(im_sp, im_lp, im_pl, features, spline_plir=spline_plir, spline_plc=spline_nf)
    ip.print_metrics(features)
    log = np.log(features['im_tau_bulk_f32'])
    view = ImageViewer(features['im_tau_bulk_f32'])
    #ImageViewer(log)
    view.show()


def run_plir2():
    fn = r"C:\Users\Neil\BT\Data\2017-09-06 TransferFunctions.TXT"
    vals = features_block.load_transfer(fn)
    spline_plir, spline_nf, spline_sp, spline_lp = features_block.interpolate_transfer(vals, debug=False)

    if False:
        fn_sp = r"C:\Users\Neil\BT\Data\blocks\PLIR\2017-11-01\plg.meas.block.b3bl.north.sp.img.tif"
        fn_lp = r"C:\Users\Neil\BT\Data\blocks\PLIR\2017-11-01\plg.meas.block.b3bl.north.lp.img.tif"
    elif False:
        fn_sp = r"C:\Users\Neil\Desktop\1172\plg.meas.block.b3bl.north.sp.img.tif"
        fn_lp = r"C:\Users\Neil\Desktop\1172\plg.meas.block.b3bl.north.lp.img.tif"
    else:
        fn_sp = r"C:\Users\Neil\Desktop\Rietech.2.1172\tifs\plg.meas.block.b3bl.north.sp.img.tif"
        fn_lp = r"C:\Users\Neil\Desktop\Rietech.2.1172\tifs\plg.meas.block.b3bl.north.lp.img.tif"

    im_sp = ip.open_image(fn_sp).astype(np.float32)
    im_lp = ip.open_image(fn_lp).astype(np.float32)

    if False:
        im_sp = ndimage.zoom(im_sp, zoom=0.5)
        im_lp = ndimage.zoom(im_lp, zoom=0.5)

    features = {}
    features_block.plir2(im_sp, im_lp, features, spline_plir=spline_plir, spline_sp=spline_sp)
    ip.print_metrics(features)
    log = np.log(features['im_tau_bulk_f32'])
    view = ImageViewer(features['im_tau_bulk_f32'])
    ImageViewer(log)
    plt.figure()
    plt.plot(features['im_tau_bulk_f32'].mean(axis=0))
    view.show()


def run_block():
    fn = r"C:\Users\Neil\BT\Data\blocks\misc\brick JW - Test PL Image %28PL Image%29.tif"
    #fn = r"C:\Users\Neil\BT\Data\blocks\B4\691 - PL Image B4 N2 4V (PL Image - Composite).tif"
    #fn = r"C:\Users\Neil\BT\Data\blocks\P3045564-20 ten times\.tif"
    #fn = r"C:\Users\Neil\BT\Data\blocks\P3045564-20 ten times\427 - P3045564-20-1 (PL Image).tif"
    im_pl = ip.open_image(fn).astype(np.float32)
    features = {}
    features_block.feature_extraction(im_pl, features)
    rgb = features_block.create_overlay(features)
    ip.print_metrics(features)
    view = ImageViewer(im_pl)
    ImageViewer(rgb)
    view.show()


def main():
    if False:
        run_block()
    elif False:
        run_plir2()
    else:
        run_plir()


if __name__ == '__main__':
    main()
