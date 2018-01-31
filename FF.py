import os
import sys
from features_multi_wafer import WaferType
import cPickle as pickle
from image_processing import ImageViewer
import image_processing as ip
import cropping
import numpy as np
from datetime import datetime
import pandas as pd
from scipy import ndimage
import glob
from numpy.fft import ifftshift, fftshift
import cv2
import timeit
from skimage import draw
import pixel_ops
import features_multi_wafer as fe

# sample every SAMPLE_RATE'th middle wafer
SAMPLE_RATE = 100

# recompute the FF correction after RECOMPUTE_RATE samples
#  NOTE: the FF we be recomputed after a minimum of SAMPLE_RATE*RECOMPUTE_RATE wafers
RECOMPUTE_RATE = 100


def load_stats():
    with open(fn_count, 'rb') as f:
        return pickle.load(f)


def save_stats(stats):
    with open(fn_count, 'wb') as f:
        pickle.dump(stats, f, protocol=0)


folder_ff = os.path.join(os.getcwd(), 'FF')
if not os.path.isdir(folder_ff):
    os.makedirs(folder_ff)
fn_count = os.path.join(folder_ff, "ff_stats.pkl")
if not os.path.isfile(fn_count):
    stats_empty = {'middle_count': 0, 'sample_count': 0}
    save_stats(stats_empty)


def compute_batch_correction(fn_samples, debug=False):
    df = pd.read_csv(fn_samples)
    if not debug and len(df) != RECOMPUTE_RATE:
        print "Incomplete file"
        return None

    # find the foreground top and bottom
    top = int(round(np.median(df.ix[:, 'top'])))
    bottom = int(round(np.median(df.ix[:, 'bottom'])))

    # average and smooth FF
    ff_data = df.ix[:, 'row_0':].values.astype(np.float32)
    ff_norm = ff_data / np.c_[np.median(ff_data, axis=1)]

    if False:
        # import matplotlib.pylab as plt
        # plt.figure()
        # plt.imshow(ff_norm, interpolation="nearest")
        # plt.show()
        view = ImageViewer(ff_data)
        view = ImageViewer(ff_norm)
        view.show()

    # ff_avg = np.median(ff_norm, axis=0)
    ff_avg = np.mean(ff_norm, axis=0)
    ff_avg[:top] = ff_avg[top]
    ff_avg[bottom:] = ff_avg[bottom]
    sigma = 0.005 * len(ff_avg)
    ff_smooth = ndimage.gaussian_filter1d(ff_avg, sigma=sigma)
    correction = 1.0 / ff_smooth

    if False:
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(ff_smooth)
        plt.plot(ff_smooth * correction)
        plt.show()

    return correction


def compute_ff_correction(im, crop_props, wafer_features, fn=""):
    if wafer_features['wafer_type'] != WaferType.MIDDLE:
        return

    stats = load_stats()

    if stats['middle_count'] % SAMPLE_RATE == 0:
        # compute FF profile for this wafer

        # find the left and right bounds of the wafer
        s = int(round(0.02 * im.shape[0]))
        left = max(crop_props['corners'][0][1], crop_props['corners'][3][1]) + s
        right = min(crop_props['corners'][1][1], crop_props['corners'][2][1]) - s
        top = max(crop_props['corners'][0][0], crop_props['corners'][1][0]) + s
        bottom = min(crop_props['corners'][2][0], crop_props['corners'][3][0]) - s
        im_cropped = im[:, left:right]
        if False:
            view = ImageViewer(im_cropped)
            view.show()

        # get the y-profile
        y_profile = np.median(im_cropped, axis=1)
        if False:
            print top, bottom
            import matplotlib.pylab as plt
            plt.figure()
            plt.plot(y_profile)
            plt.show()

        # store profile
        fn_samples = os.path.join(folder_ff, "ffs_%04d.csv" % (stats['sample_count'] // RECOMPUTE_RATE))
        data = "%s,%s,%d,%d," % (str(datetime.now()), fn, top, bottom)
        data += ','.join(['%d' % (m) for m in y_profile.astype(np.uint16)])
        if not os.path.isfile(fn_samples):
            headers = "time,filename,top,bottom,"
            headers += ','.join(['row_%d' % (r) for r in range(len(y_profile))])
            with open(fn_samples, 'a') as f:
                f.write(headers + '\n')
        with open(fn_samples, 'a') as f:
            f.write(data + '\n')

        # compute FF correction
        if stats['sample_count'] > 0 and stats['sample_count'] % RECOMPUTE_RATE == 0:
            fn_samples = os.path.join(folder_ff, "ffs_%04d.csv" % ((stats['sample_count'] - 1) // RECOMPUTE_RATE))
            correction = compute_batch_correction(fn_samples)

            fn_correction = os.path.join(folder_ff, "ff_correction.data")
            # np.save(fn_correction, correction)
            with open(fn_correction, 'w') as f:
                vals = '\n'.join(['%0.04f' % (val) for val in correction])
                f.write(vals)

        # increment sample count
        stats['sample_count'] += 1

    # increment how many middles we've seen since
    stats['middle_count'] += 1
    save_stats(stats)

    return


def apply_ff_correction(im_orig):
    fn_correction = os.path.join(folder_ff, "ff_correction.data")
    if not os.path.isfile(fn_correction):
        print "WARNING: FF_CORRECTION is on, but no correction file found"
        return im_orig

    with open(fn_correction) as f:
        correction = np.array([float(val.strip()) for val in f.readlines()], np.float32)
    # correction = np.load(fn_correction)

    if False:
        import matplotlib.pylab as plt
        plt.figure()
        plt.plot(correction)
        plt.show()

    if im_orig.shape[0] != len(correction):
        print "ERROR: Correction resolution does not match file resolution"
        return im_orig

    im = (im_orig * np.c_[correction]).astype(im_orig.dtype)

    return im


def correct_hash_pattern(im):
    mask_peaks = np.load("hash_fft_mask.npy")
    assert im.shape == mask_peaks.shape
    h, w = im.shape

    im_type = im.dtype
    im = im.astype(np.float32)

    fft = fftshift(cv2.dft(im, flags=cv2.DFT_COMPLEX_OUTPUT))
    fft_mag = cv2.magnitude(fft[:, :, 0], fft[:, :, 1])
    fft_view = cv2.log(fft_mag)

    if False:
        view = ImageViewer(im)
        view = ImageViewer(fft_view)
        view.show()

    # create a mask for the + pattern (to prevent ringing at edges)
    # find rotation using cropping
    try:
        crop_props = cropping.crop_wafer(im, create_mask=True)
        foreground_mask = crop_props['mask']
    except:
        print("WARNING: Crop failed")
        foreground_mask = np.ones_like(im, np.uint8)

    mask_edges = np.zeros_like(im, np.float32)
    T = 2
    mask_edges[h // 2 - T:h // 2 + T + 1, :] = 1
    mask_edges[:, w // 2 - T:w // 2 + T + 1] = 1
    RADIUS = 35
    ys, xs = draw.circle(h // 2, w // 2, RADIUS)
    mask_edges[ys, xs] = 1
    if False:
        angle = -crop_props['estimated_rotation']
        rot_mat = cv2.getRotationMatrix2D((h // 2, w // 2), angle, 1.0)
        rotated = cv2.warpAffine(mask_edges, rot_mat, (mask_edges.shape[1], mask_edges.shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
    else:
        rotated = mask_edges

    mask_edges = rotated > 0.5
    mask_background = ~mask_peaks & ~mask_edges
    mask_pattern = mask_peaks & (~mask_edges)

    if False:
        view = ImageViewer(fft_view)
        view = ImageViewer(mask_peaks)
        view = ImageViewer(mask_edges)
        view = ImageViewer(mask_pattern)
        view.show()

    background_fill = fft_mag[mask_background].mean()

    fft_corrected = fft.copy()
    fft_corrected[mask_pattern, :] = (background_fill, 0)

    if False:
        fft_view_corrected = cv2.log(cv2.magnitude(fft_corrected[:, :, 0], fft_corrected[:, :, 1]))
        view = ImageViewer(fft_view)
        view = ImageViewer(fft_view_corrected)
        view.show()

    # inverse FFT
    im_corrected = cv2.idft(ifftshift(fft_corrected), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    # fill background with original values
    # im_corrected crop_props['mask']
    pixel_ops.CopyMaskF32(im, im_corrected, foreground_mask, 0)

    if False:
        view = ImageViewer(im)
        view = ImageViewer(im_corrected)
        view = ImageViewer(np.abs(im - im_corrected))
        view.show()

    return im_corrected.astype(im_type)


def compute_hash_pattern_correction(folder):
    fns = glob.glob(os.path.join(folder, "*.tif*"))

    if len(fns) == 0:
        print "No tif files found in: %s" % (folder)
        sys.exit()

    if True:
        ims = [ip.open_image(fn).astype(np.float32) for fn in fns]
        im_mean = ims[0].copy()
        for im in ims[1:]:
            im_mean += im
        im_mean /= len(ims)
        background = cv2.GaussianBlur(im_mean, (0, 0), 8, borderType=cv2.BORDER_REPLICATE)
        pattern = im_mean - background
        pattern -= pattern.mean()
    else:
        background = ip.open_image(r"C:\Users\Neil\BT\Data\R2 FFT\FF Wafer Images\precomputed\std - ff.tif").astype(
            np.float32) / 4.0
        im_mean = ip.open_image(r"C:\Users\Neil\BT\Data\R2 FFT\FF Wafer Images\precomputed\SUM_Stack.tif").astype(
            np.float32) / 4.0
        pattern = im_mean - background
        pattern -= pattern.mean()

    if False:
        view = ImageViewer(im_mean)
        ImageViewer(background)
        ImageViewer(pattern)
        view.show()
        sys.exit()

    # find a mask of the peaks
    fft = fftshift(cv2.dft(pattern, flags=cv2.DFT_COMPLEX_OUTPUT))
    fft_mag = cv2.magnitude(fft[:, :, 0], fft[:, :, 1])
    fft_smooth = cv2.GaussianBlur(cv2.medianBlur(fft_mag, ksize=5), ksize=(0, 0), sigmaX=5)
    fft_log = cv2.log(fft_smooth)
    THRESH = 13.75
    mask = fft_log > THRESH

    # ignore middle (low frequency stuff)
    RADIUS = 35

    h, w = pattern.shape
    ys, xs = draw.circle(h // 2, w // 2, RADIUS)
    mask[ys, xs] = 0

    np.save("hash_fft_mask.npy", mask)
    print "FFT mask saved to 'hash_fft_mask.npy'"

    if False:
        view = ImageViewer(fft_log)
        view = ImageViewer(mask)
        view.show()


def correct_waffle(im):
    # load FFT of
    fn = "fft_hash_pattern.npy"
    if os.path.isfile(fn):
        fft_pattern = np.load(fn)
    else:
        # isolate waffle pattern
        ff = ip.open_image(r"C:\Users\Neil\Dropbox (Personal)\BT\Data\R2 FFT\FF Wafer Images\std - ff.tif").astype(
            np.float32) / 4.0
        stack = ip.open_image(r"C:\Users\Neil\Dropbox (Personal)\BT\Data\R2 FFT\FF Wafer Images\SUM_Stack.tif").astype(
            np.float32) / 4.0
        pattern = stack - ff
        pattern -= pattern.mean()
        pattern /= pattern.std()

        # FFT
        fft_pattern = fftshift(cv2.dft(pattern, flags=cv2.DFT_COMPLEX_OUTPUT))

        # smooth
        fft_pattern_mag = cv2.magnitude(fft_pattern[:, :, 0], fft_pattern[:, :, 1])
        fft_pattern_smooth = cv2.GaussianBlur(cv2.medianBlur(fft_pattern_mag, ksize=5), ksize=(0, 0), sigmaX=5)
        fft_pattern = cv2.log(fft_pattern_smooth)

        # remove non-peaks
        fft_pattern -= np.mean(fft_pattern)
        fft_pattern[fft_pattern < 0] = 0

        np.save(fn, fft_pattern)

    # mask for middle '+'
    mask_edges = np.zeros_like(fft_pattern, np.bool)
    T = 2
    h, w = mask_edges.shape
    mask_edges[h // 2 - T:h // 2 + T + 1, :] = True
    mask_edges[:, w // 2 - T:w // 2 + T + 1] = True
    RADIUS = 50
    ys, xs = draw.circle(h // 2, w // 2, RADIUS)
    mask_edges[ys, xs] = True

    if False:
        # view = ImageViewer(fft_pattern_unmasked)
        view = ImageViewer(fft_pattern)
        view.show()
        sys.exit()

    # fft of wafer image
    fft = fftshift(cv2.dft(im, flags=cv2.DFT_COMPLEX_OUTPUT))
    fft_mag = cv2.magnitude(fft[:, :, 0], fft[:, :, 1])
    fft_phase = cv2.phase(fft[:, :, 0], fft[:, :, 1])
    fft_log = cv2.log(fft_mag)
    fft_wafer_smooth = cv2.GaussianBlur(cv2.medianBlur(fft_log, ksize=5), ksize=(0, 0), sigmaX=5)

    if True:
        view = ImageViewer(fft_pattern)
        view = ImageViewer(fft_wafer_smooth)
        view.show()
        sys.exit()

    # find fit between background of waffle FFT and wafer FFT
    # 1. fit background
    background_mask = ((fft_pattern == 0) & (~mask_edges))
    peak_mask = ((fft_pattern > 0.2) & (~mask_edges))

    if False:
        view = ImageViewer(background_mask)
        view = ImageViewer(peak_mask)
        view.show()
        sys.exit()

    # 2. fit peaks
    pattern_vals = fft_pattern[peak_mask]
    wafer_vals = fft_wafer_smooth[peak_mask]

    def dist(params, pattern_vals, wafer_vals):
        shift, scale = params

        pattern_vals_fit = (pattern_vals * scale) + shift

        return ((pattern_vals_fit - wafer_vals) ** 2).mean()

    from scipy import optimize
    params = (wafer_vals.mean(), 1)
    t1 = timeit.default_timer()
    shift, scale = optimize.fmin(dist, params, args=(pattern_vals, wafer_vals))
    t2 = timeit.default_timer()

    if False:
        print "Optimization time: ", t2 - t1
        print shift, scale
        fft_fit = (fft_pattern * scale) + shift
        vmin = min(fft_fit.min(), fft_wafer_smooth.min())
        vmax = max(fft_fit.max(), fft_wafer_smooth.max())

        view = ImageViewer(fft_fit, vmin=vmin, vmax=vmax)
        view = ImageViewer(fft_wafer_smooth, vmin=vmin, vmax=vmax)
        view.show()
        sys.exit()

    # apply correction
    correction = fft_pattern * -scale
    correction[mask_edges] = 0
    corrected_log = fft_log + correction
    corrected_mag = np.e ** corrected_log
    fft_real = np.cos(fft_phase) * corrected_mag
    fft_imag = np.sin(fft_phase) * corrected_mag
    fft_corrected = np.dstack((fft_real, fft_imag))
    im_corrected = cv2.idft(ifftshift(fft_corrected), flags=cv2.DFT_REAL_OUTPUT | cv2.DFT_SCALE)

    if False:
        view = ImageViewer(im)
        view = ImageViewer(im_corrected)
        view.show()
        sys.exit()

    # create a mask for the + pattern (to prevent ringing at edges)
    # find rotation using cropping
    if True:
        try:
            crop_props = cropping.crop_wafer(im, create_mask=True)
            pixel_ops.CopyMaskF32(im, im_corrected, crop_props['mask'], 0)
        except:
            print("WARNING: Crop failed")
            return im

    return im_corrected


def main():
    pass


if __name__ == '__main__':
    main()
