import sys
import os
import numpy as np
from scipy import ndimage, stats
import TIFFfile
import PIL.Image
import io
import pixel_ops
import cc_label
import math
import cv2
import fnmatch
import matplotlib.pylab as plt
from pprint import pprint


def recursive_glob(folder, wildcard):
    matches = []
    for root, dirnames, filenames in os.walk(folder):
        for filename in fnmatch.filter(filenames, wildcard):
            matches.append(os.path.join(root, filename))

    return matches


thinning_lut = np.array([0, 0, 0, 1, 0, 0, 1, 3, 0, 0, 3, 1, 1, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 2, 0, 3, 0, 3, 3,
                         0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 3, 0, 2, 2,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 2, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 2, 0,
                         0, 0, 3, 1, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                         3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         2, 3, 1, 3, 0, 0, 1, 3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         2, 3, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 1, 0, 0, 0, 0, 2, 2, 0, 0, 2, 0, 0,
                         0], np.uint8)


def open_image(fn, rgb2gray=True, cast_long=True):
    im = None
    if os.path.splitext(fn)[1].lower() in ['.tif', '.tiff']:
        im = TIFFfile.imread(fn)
        if cast_long: im = im.astype(np.uint16)
    else:
        im = PIL.Image.open(fn)
        im = np.array(im)
    if im.ndim == 3 and rgb2gray:
        print('WARNING: 3-channel image. Using first channel.')
        im = np.ascontiguousarray(im[:, :, 0])

    return im


def ImageViewer(im, cmap=None, interpolation='nearest', **kwargs):
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if type(im) is str: im = open_image(im, rgb2gray=False)

    if cmap is None: cmap = cm.gray
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(im, interpolation=interpolation, cmap=cmap, **kwargs)

    h = im.shape[0]
    w = im.shape[1]
    if im.ndim == 2:
        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < w and row >= 0 and row < h:
                z = im[row, col]
                return '%1.3f (y=%1.1f, x=%1.1f)' % (z, y, x)
            else:
                return 'x=%1.1f, y=%1.1f' % (x, y)
    else:
        def format_coord(x, y):
            col = int(x + 0.5)
            row = int(y + 0.5)
            if col >= 0 and col < w and row >= 0 and row < h:
                r = im[row, col, 0]
                g = im[row, col, 1]
                b = im[row, col, 2]
                return 'y=%1.1f, x=%1.1f, R=%d, G=%d, B=%d' % (y, x, r, g, b)
            else:
                return 'x=%1.1f, y=%1.1f' % (x, y)

    ax.format_coord = format_coord
    plt.autoscale(tight=True)

    return plt


def histogram_percentiles(im, features, center_y=None, center_x=None, radius=None, skip_zero=False):
    num_pixels = im.shape[0] * im.shape[1]

    # compute smooth histogram
    hist_int = np.zeros(2 ** 16, np.int32)
    if center_y is None:
        pixel_ops.FastHistogram(im, hist_int)
    else:
        pixel_ops.FastHistogramDisc(im, hist_int, center_y, center_x, radius)

    max_pl = min(np.nonzero(hist_int)[0][-1] + 100, 2 ** 16)
    hist = hist_int[:max_pl]
    hist = np.ascontiguousarray(hist[:max_pl].astype(np.float32))
    if skip_zero:
        hist[0] = 0

    # find a suitable smoothing factor for pixels in this range
    im_mean = cv2.mean(im)[0]
    sigma = im_mean / 75.0

    # smooth
    sigma = max(1.0, sigma)
    smoothed_hist = ndimage.gaussian_filter1d(hist, sigma=sigma, mode="constant", cval=0)

    # find most common PL
    peak_pl = np.argmax(smoothed_hist)

    # find moments of distribution around peak
    #  - compute using histogram for speed
    bins = np.arange(max_pl).astype(np.float64)
    variance = (((bins - peak_pl) ** 2) * hist).sum() / float(num_pixels)
    skew = (((((bins - peak_pl) ** 3) * hist).sum() / float(num_pixels)) / math.pow(variance, 1.5))

    # range (1st and 99th percentile to ignore outliers)
    hist_cdf = np.cumsum(smoothed_hist) / smoothed_hist.sum()
    percentile_01 = np.abs(hist_cdf - 0.01).argmin()
    percentile_90 = np.abs(hist_cdf - 0.90).argmin()
    percentile_95 = np.abs(hist_cdf - 0.95).argmin()
    percentile_99 = np.abs(hist_cdf - 0.99).argmin()
    percentile_999 = np.abs(hist_cdf - 0.9975).argmin()

    features['hist_peak'] = peak_pl
    features['hist_std'] = math.sqrt(variance)
    features['hist_cov'] = features['hist_std'] / max(1, im_mean)
    features['hist_skew'] = skew
    features['hist_percentile_01'] = percentile_01
    features['hist_median'] = np.abs(hist_cdf - 0.50).argmin()
    features['hist_percentile_90'] = percentile_90
    features['hist_percentile_95'] = percentile_95
    features['hist_percentile_99'] = percentile_99
    features['hist_percentile_99.9'] = percentile_999
    features['hist_mean'] = im_mean

    features["hist_harmonic_mean"] = 1.0 / (1.0 / np.maximum(0.01, im.flat)).mean()

    if False:
        import matplotlib.pylab as plt
        # print features['hist_percentile_01']
        # print features['hist_peak']
        print features['hist_percentile_99.9'] - features['hist_percentile_95']
        print features['hist_percentile_95']
        plt.figure()
        plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest')
        plt.figure()
        plt.plot(smoothed_hist)
        plt.figure()
        plt.plot(hist)
        plt.show()
        # sys.exit()

    return smoothed_hist


def closest_local_max(signal, point):
    local_maxs = np.where((signal > np.roll(signal, 1)) & (signal > np.roll(signal, -1)))[0]
    local_max = local_maxs[np.argmin(np.abs(local_maxs - point))]

    if False:
        plt.figure()
        plt.plot(signal)
        plt.plot(local_max, signal[local_max], 'o')
        plt.vlines([point], signal.min(), signal.max())
        plt.show()

    return local_max


# @profile
def fast_percentile(im, filer_size, percentile):
    down_size = 8
    small = ndimage.zoom(im, zoom=(1.0 / down_size), order=1, mode="nearest")

    if False:
        view = ImageViewer(im)
        view = ImageViewer(small)
        view.show()
        sys.exit()

    mf = ndimage.percentile_filter(small, percentile=percentile, size=filer_size, mode='nearest')

    if False:
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]] / float(down_size)
        coords = np.empty(np.r_[2, im.shape], dtype=float)
        coords[0, ...] = Y
        coords[1, ...] = X
        return ndimage.interpolation.map_coordinates(mf, coordinates=coords, order=1, mode='nearest')
    else:
        orig_size = ndimage.zoom(mf, zoom=down_size, order=1, mode="nearest")
        if orig_size.shape[0] < im.shape[0]:
            orig_size = np.vstack((orig_size, np.repeat(orig_size[-1, :].reshape(1, orig_size.shape[1]),
                                                        repeats=(im.shape[0] - orig_size.shape[0]), axis=0)))
        if orig_size.shape[1] < im.shape[1]:
            orig_size = np.hstack((orig_size, np.repeat(orig_size[:, -1].reshape(orig_size.shape[0], 1),
                                                        repeats=(im.shape[1] - orig_size.shape[1]), axis=1)))
        return orig_size


def get_percentile(im, percentile):
    vals = sorted(im[::3, ::3].flat)
    num_pixels = len(vals)
    i_lower = int(percentile * num_pixels)
    i_upper = int((1 - percentile) * num_pixels)
    lower_bound = vals[i_lower]
    upper_bound = vals[i_upper]

    return lower_bound, upper_bound


def trim_percentile(im, percentile):
    # clip some values (in case a little background is showing)
    # downsample first
    vals = sorted(im[::3, ::3].flat)
    num_pixels = len(vals)
    i_lower = int(percentile * num_pixels)
    i_upper = int((1 - percentile) * num_pixels)
    lower_bound = vals[i_lower]
    upper_bound = vals[i_upper]

    return np.clip(im, lower_bound, upper_bound, im)


def scale_image(im):
    """
    Scale an image to the range [0, 1]
    """
    dynamic_range = im.max() - im.min()

    if dynamic_range > 0:
        return (im - im.min()) / float(dynamic_range)
    else:
        return np.zeros_like(im)


def save_image(fnOut, im, scale=True):
    if im.dtype.type is np.uint8:
        if im.ndim == 2 and scale:
            im = (scale_image(im.astype(np.float64)) * 255).astype(np.uint8)
        pil_im = PIL.Image.fromarray(im)
        pil_im.save(fnOut)
    elif im.dtype.type is np.uint16:
        TIFFfile.imsave(fnOut, im)
    elif im.dtype.type is np.float32 and im.ndim == 2:
        im = (scale_image(im) * 255).astype(np.uint8)
        pil_im = PIL.Image.fromarray(im)
        pil_im.save(fnOut)
    else:
        print "ERROR: Unsupported file type. Type: %s  Dim: %d  Filename: %s" % (str(im.dtype.type), im.ndim, fnOut)
        sys.exit()


def overlay_mask(im, mask, colour='g', rgb=None):
    if im.ndim == 2:
        im = (scale_image(im.astype(np.float64)) * 255).astype(np.uint8)
        rgb_im = np.dstack((im, im, im))
    elif im.ndim == 3:
        rgb_im = im

    if rgb == None:
        if colour == 'g':
            rgb = (0, 128, 0)
        elif colour == 'r':
            rgb = (200, 0, 0)
        elif colour == 'b':
            rgb = (0, 0, 128)
        elif colour == 'w':
            rgb = (255, 255, 255)

    rgb_im[mask == 1, 0] = rgb[0]
    rgb_im[mask == 1, 1] = rgb[1]
    rgb_im[mask == 1, 2] = rgb[2]

    return rgb_im


def anisotropic_diffusion(im, niter, kappa, gamma):
    # initialize internal variables
    deltaS = np.empty_like(im)
    deltaE = np.empty_like(im)
    gS = np.empty_like(im)
    gE = np.empty_like(im)
    NS = np.empty_like(im)
    EW = np.empty_like(im)

    kappa_sqr = kappa ** 2

    for ii in range(niter):
        # calculate the diffs
        deltaS[:-1, :] = np.diff(im, axis=0)
        deltaE[:, :-1] = np.diff(im, axis=1)

        pixel_ops.cal_img_grad(deltaS, gS, kappa_sqr)
        pixel_ops.cal_img_grad(deltaE, gE, kappa_sqr)

        # update matrices
        E = gE * deltaE
        S = gS * deltaS

        # subtract a copy that has been shifted 'North/West' by one pixel.
        NS[:] = S
        EW[:] = E
        NS[1:, :] -= S[:-1, :]
        EW[:, 1:] -= E[:, :-1]

        # update the image
        pixel_ops.update_img_grad(im, NS, EW, gamma)

    return im


def connected_components(bw):
    max_labels = (bw.shape[0] * bw.shape[1]) // 4
    list1 = np.zeros(max_labels, np.int32)
    list2 = np.zeros(max_labels, np.int32)
    list3 = np.zeros(max_labels, np.int32)

    z0 = np.zeros((bw.shape[0], 2), np.uint8)
    padded = np.hstack((z0, bw, z0))
    z1 = np.zeros((2, padded.shape[1]), np.uint8)
    padded = np.vstack((z1, padded, z1))
    ccs = np.zeros_like(padded, np.int32)
    num_ccs = cc_label.CCLabel(padded, ccs, list1, list2, list3)
    ccs = np.ascontiguousarray(ccs[2:-2, 2:-2])

    return ccs, num_ccs


def remove_small_ccs(im, size_thresh):
    ccs, num_ccs = connected_components(im)
    line_sizes = np.zeros(num_ccs + 1, np.int32)
    pixel_ops.CCSizes(ccs, line_sizes)
    # print line_sizes
    pixel_ops.RemoveSmallCCs(im, ccs, line_sizes, size_thresh)


def fast_smooth(im, sigma, pad_mode="edge"):
    h, w = im.shape

    if pad_mode == "edge":
        borderType = cv2.BORDER_REPLICATE
    elif pad_mode == "reflect":
        borderType = cv2.BORDER_REFLECT

    im_down = cv2.resize(im, (w // 2, h // 2))
    smooth = cv2.GaussianBlur(im_down, (0, 0), sigma / 2.0, borderType=borderType)
    smooth = cv2.resize(smooth, (w, h))

    return smooth


def polar_transform(im, center, num_angles=None, max_radius=None,
                    interp="nearest"):  # angles=None, radii=None, center=None):
    """Return polar transformed image."""
    h, w = im.shape
    w2 = (w // 2)

    if num_angles is None:
        num_angles = h

    if max_radius is None:
        max_radius = int(round(math.sqrt(2 * (w2 ** 2))))

    theta = np.empty((num_angles, max_radius), dtype=np.float64)
    theta.T[:] = -np.linspace(0, 2 * np.pi, num_angles, endpoint=False)
    radius = np.empty_like(theta)
    radius[:] = np.arange(max_radius, dtype=np.float64)  # / 2.0

    Y = radius * np.sin(theta) + center[0]
    X = radius * np.cos(theta) + center[1]

    if False:
        view = ImageViewer(im)
        print radius.min(), radius.max()
        print theta.min(), theta.max()
        ImageViewer(radius)
        ImageViewer(theta)
        ImageViewer(Y)
        ImageViewer(X)
        view.show()
        sys.exit()

    return ndimage.interpolation.map_coordinates(im, [Y, X], order=1, mode=interp)


def polar_transform_inverse(im, pt, center):  # angles=None, radii=None, center=None):
    """Return inverse polar transformed image."""
    (h, w) = im.shape

    angles = np.empty((h, w), dtype=np.float64)
    radii = np.empty_like(angles)
    pixel_ops.CenterDistanceF64(radii, angles, center[0], center[1])
    angles[angles < 0.0] += 2.0 * math.pi

    if False:
        view = ImageViewer(im)
        view = ImageViewer(pt)
        print radii.min(), radii.max()
        print angles.min(), angles.max()
        ImageViewer(angles)
        ImageViewer(radii)
        view.show()
        sys.exit()

    Y = np.empty((h, w), dtype=np.float64)
    X = np.empty_like(Y)
    Y[:] = (pt.shape[0] - 1) - ((pt.shape[0] - 1) * (angles / (2 * np.pi)))
    X[:] = pt.shape[1] * (radii / ((pt.shape[1] - 1)))

    if False:
        view = ImageViewer(im)
        ImageViewer(angles)
        ImageViewer(radii)
        ImageViewer(Y)
        ImageViewer(X)
        view.show()
        sys.exit()

    return ndimage.interpolation.map_coordinates(pt, [Y, X], order=1)


################
# min_rectanlge #
################
def min_rectanlge(cv_hull):
    """
    Find the minimum enclosing rectangle of a set of points
    TODO: check if opencv has a better and/or faster implementation
    """

    def max_dist(j, n, s, c, mx, my):  # advance j to extreme point
        xn, yn = cv_hull[j][0], cv_hull[j][1]
        rx, ry = xn * c - yn * s, xn * s + yn * c
        best = mx * rx + my * ry
        while True:
            x, y = rx, ry
            xn, yn = cv_hull[(j + 1) % n][0], cv_hull[(j + 1) % n][1]
            rx, ry = xn * c - yn * s, xn * s + yn * c
            if mx * rx + my * ry >= best:
                j = (j + 1) % n
                best = mx * rx + my * ry
            else:
                return (x, y, j)

    n = len(cv_hull)
    iL = iR = iP = 1  # indexes left, right, opposite
    pi = 4 * math.atan(1)
    minRect = (1e33, 0, 0, 0, 0, 0, 0, 0)  # area, dx, dy, i, iL, iP, iR
    for i in range(n - 1):
        dx = cv_hull[i + 1][0] - cv_hull[i][0]
        dy = cv_hull[i + 1][1] - cv_hull[i][1]
        theta = pi - math.atan2(dy, dx)
        s, c = math.sin(theta), math.cos(theta)
        yC = cv_hull[i][0] * s + cv_hull[i][1] * c

        xP, yP, iP = max_dist(iP, n, s, c, 0, 1)
        if i == 0: iR = iP
        xR, yR, iR = max_dist(iR, n, s, c, 1, 0)
        xL, yL, iL = max_dist(iL, n, s, c, -1, 0)
        area = (yP - yC) * (xR - xL)
        if area < minRect[0]:
            minRect = (area, xR - xL, yP - yC, i, iL, iP, iR, theta)

    if False:
        print minRect[:-1]
        print math.degrees(minRect[-1])
        import matplotlib.pylab as plt
        plt.figure()
        hull2 = np.array(cv_hull)
        plt.scatter(hull2[:, 0], hull2[:, 1])
        plt.scatter(hull2[minRect[3:-1], 0], hull2[minRect[3:-1], 1], color="r")
        plt.show()
        sys.exit()

    length = max(minRect[1], minRect[2])
    width = min(minRect[1], minRect[2])
    theta = math.degrees(minRect[-1])
    if theta > 180:
        theta -= 180

    return length, width, theta


def encode_png(im):
    pil_im = PIL.Image.fromarray(im)
    png = io.BytesIO()
    pil_im.save(png, format="PNG")

    return png.getvalue()


def decode_png(data):
    fh = io.BytesIO(data)
    pil_im = PIL.Image.open(fh)

    return np.array(pil_im)


def print_metrics(features, display=True):
    f = features.copy()
    for k in f.keys():
        if k in ['_fn']:
            continue

        try:
            _ = float(f[k])
        except:
            del f[k]
    if display:
        pprint(f)

    return f


def list_images(features):
    im_names = []
    for k in features.keys():
        if k.split('_')[-1] not in ['u8', 'u16', 'f32'] or k[0] == '_':
            continue
        im_names.append(k)

    return im_names


if __name__ == '__main__':
    pass
