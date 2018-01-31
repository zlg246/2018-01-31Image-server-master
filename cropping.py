import os, sys
import numpy as np
import image_processing as ip
from image_processing import ImageViewer
import cv2
from scipy import ndimage, optimize, stats
from scipy.interpolate import interp1d
import parameters
import math
import pixel_ops
import matplotlib.pylab as plt
import matplotlib.cm as cm
import timeit
import PIL.Image
import PIL.ImageDraw as ImageDraw
from skimage.morphology import convex_hull_image
import traceback
from skimage import draw
from pprint import pprint
import cell_processing as cell


class WaferMiddleException(Exception):
    pass


class WaferMissingException(Exception):
    pass


class CellMissingException(Exception):
    pass


def correct_rotation(im, crop_props, pad=True, border_erode=parameters.BORDER_ERODE, fix_chamfer=True):
    # correct rotation
    angle = crop_props['estimated_rotation']
    cy, cx = crop_props['center']
    rot_mat = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    rotated = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    assert im.shape == rotated.shape
    h, w = im.shape

    if False:
        view = ImageViewer(im)
        ImageViewer(rotated)
        plt.show()
        sys.exit()

    theta = np.deg2rad(angle) * -1
    tl = crop_props['corners_floats'][0]
    br = crop_props['corners_floats'][2]
    ps = [tl, br]
    tlbr = []
    for p in ps:
        [y, x] = p
        # coordinates in image
        y_c = (y - cy)
        x_c = (x - cx)
        y_r = ((x_c * math.sin(theta)) + (y_c * math.cos(theta)))
        x_r = ((x_c * math.cos(theta)) - (y_c * math.sin(theta)))
        y_r += cy
        x_r += cx
        tlbr.append(y_r)
        tlbr.append(x_r)

    t = int(round(tlbr[0] + border_erode)) + 1
    l = int(round(tlbr[1] + border_erode)) + 1
    b = int(round(tlbr[2] - border_erode))
    r = int(round(tlbr[3] - border_erode))

    cropped = rotated[max(0, t):min(b, h - 1), max(l, 0):min(r, w - 1)]

    if fix_chamfer:
        # due to chamfer, the corners of a wafer may not be perfectly square,
        #  and this can cause a problem if a low value for border erosion is
        #  used. In particular, a small amount of background can be visible,
        #  and the low values can cause false impure regions at the corners.
        #  To address this, replace pixels values in the corners with local
        #  median
        cropped[:4, :4] = cropped[2:4, 2:4].mean()
        cropped[-4:, :4] = cropped[-4:-2, 2:4].mean()
        cropped[:4, -4:] = cropped[2:4, -4:-2].mean()
        cropped[-4:, -4:] = cropped[-4:-2, -4:-2].mean()

    if False:
        print max(0, t), min(b, h - 1), max(l, 0), min(r, w - 1)
        print cropped.shape
        view = ImageViewer(im)
        ImageViewer(rotated)
        ImageViewer(cropped)
        view.show()
        sys.exit()

    if pad:
        im_min = cropped.min()

        # if wafer is partially outside of the image, pad with 0's
        if t < 0:
            cropped = np.vstack((np.ones((abs(t), cropped.shape[1]), dtype=cropped.dtype) * im_min, cropped))
        if b > rotated.shape[0]:
            cropped = np.vstack(
                (cropped, np.ones(((b - rotated.shape[0]), cropped.shape[1]), dtype=cropped.dtype) * im_min))
        if r > rotated.shape[1]:
            cropped = np.hstack(
                (cropped, np.ones((cropped.shape[0], (r - rotated.shape[1])), dtype=cropped.dtype) * im_min))
        if l < 0:
            cropped = np.hstack((np.ones((cropped.shape[0], abs(l)), dtype=cropped.dtype) * im_min, cropped))

    return cropped


def guess_width(im, row_diff, col_diff, buff_size):
    # align the high peak from each image
    if np.argmax(row_diff) > row_diff.shape[0] / 2:
        row_diff = row_diff[::-1]
    p1 = np.argmax(row_diff)
    if np.argmax(col_diff) > col_diff.shape[0] / 2:
        col_diff = col_diff[::-1]
    p2 = np.argmax(col_diff)

    if row_diff.shape[0] < col_diff.shape[0]:
        row_diff = np.hstack((row_diff, np.zeros(col_diff.shape[0] -
                                                 row_diff.shape[0])))
    elif col_diff.shape[0] < row_diff.shape[0]:
        col_diff = np.hstack((col_diff, np.zeros(row_diff.shape[0] -
                                                 col_diff.shape[0])))

    diff_sum = row_diff + np.roll(col_diff, p1 - p2)
    w2 = diff_sum.shape[0] / 2
    wafer_left = np.argmax(diff_sum[:w2])
    wafer_right = w2 + np.argmax(diff_sum[w2:])
    wafer_width = wafer_right - wafer_left

    if False:
        print wafer_width
        plt.figure()
        plt.plot(row_diff)
        plt.figure()
        plt.plot(col_diff)
        plt.figure()
        plt.plot(diff_sum)
        plt.show()

    return wafer_width


def approx_rotation(edges):
    h, w = edges.shape
    row_diff = edges.mean(axis=1)
    col_diff = edges.mean(axis=0)

    # find the strongest opposing edges, and make sure they are horizontal
    mid = (row_diff.shape[0] + col_diff.shape[0]) // 4
    tb = min(row_diff[:mid].max(), row_diff[mid:].max())
    lr = min(col_diff[:mid].max(), col_diff[mid:].max())
    width_horizontal = True
    if lr > tb:
        edges = np.ascontiguousarray(edges.T)
        row_diff = col_diff
        width_horizontal = False
        h, w = w, h

    # first guess at top and bottom edges
    mid = row_diff.shape[0] // 2
    y1 = mid + np.argmax(row_diff[mid:])
    y2 = np.argmax(row_diff[:mid])
    ys_t = np.ones(w, np.float32) * y1
    ys_b = np.ones(w, np.float32) * y2
    xs_t = np.arange(w).astype(np.float32)
    xs_b = np.arange(w).astype(np.float32)
    ys = np.hstack((ys_t, ys_b))
    xs = np.hstack((xs_t, xs_b))

    if False:
        edges[ys.astype(np.int32), xs.astype(np.int32)] = edges.max()
        view = ImageViewer(edges)
        view.show()
        sys.exit()

    # approximate rotation
    params = (0, 0, 0, 1)
    center_y = (y1 + y2) // 2
    center_x = w // 2
    args = (center_y, center_x, ys, xs, edges, 0, h, w)
    op_params = optimize.fmin_powell(pixel_ops.WaferFit, params, args,
                                     ftol=0.1, maxfun=300, disp=False)
    rotation = math.degrees(op_params[0])

    if False:
        pixel_ops.WaferFit(op_params, center_y, center_x, ys, xs, edges, 1, h, w)
        view = ImageViewer(edges)
        view.show()
        sys.exit()

    if not width_horizontal:
        h, w = w, h
        edges = edges.T

    center_y = h // 2
    center_x = w // 2
    rot_mat = cv2.getRotationMatrix2D((center_y, center_x), -rotation, 1.0)
    rot_mat_i = cv2.getRotationMatrix2D((center_y, center_x), rotation, 1.0)
    rotated = cv2.warpAffine(edges, rot_mat, (w, h),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)

    if False:
        view = ImageViewer(edges)
        view = ImageViewer(rotated)
        view.show()
        sys.exit()

    return rot_mat, rot_mat_i, rotated, rotation


def draw_crop_box(im, crop_props, mode="pil"):
    im = im.copy().astype(np.float64)
    # draw green border
    points = crop_props['corners']
    ip.trim_percentile(im, 0.001)
    im = (ip.scale_image(im) * 255).astype(np.uint8)
    im_rgb = np.zeros((im.shape[0], im.shape[1], 3), np.uint8)
    if mode in ["pil", "rgb"]:
        im_rgb[:, :, 0] = im
        im_rgb[:, :, 1] = im
        im_rgb[:, :, 2] = im

    pil_im = PIL.Image.fromarray(im_rgb)
    draw = ImageDraw.Draw(pil_im)

    # borders
    for i in range(4):
        p0 = points[i]
        p1 = points[(i + 1) % 4]
        draw.line([(p0[1], p0[0]), (p1[1], p1[0])], fill=(0, 200, 0), width=1)

    if crop_props.has_key('radius'):
        r = crop_props['radius']
        (center_y, center_x) = crop_props['center']
    elif crop_props.has_key('wafer_radius'):
        r = crop_props['wafer_radius']
        if crop_props.has_key("_wafer_middle_orig"):
            (center_y, center_x) = crop_props['_wafer_middle_orig']
        elif crop_props.has_key("center"):
            (center_y, center_x) = crop_props['center']
        else:
            (center_y, center_x) = im.shape[0] // 2, im.shape[1] // 2
    else:
        r = None

    if r is not None:
        # box = math.sqrt((r ** 2) * 2)
        draw.ellipse([(center_x - r, center_y - r),
                      (center_x + r, center_y + r)], outline=(0, 200, 0))

    if mode == "pil":
        return pil_im
    elif mode == "rgb":
        return np.array(pil_im)
    elif mode == "mask":
        im_rgb = np.array(pil_im)
        return (im_rgb[:, :, 1] > 0).astype(np.uint8)


def crop_wafer(im_orig, width_prior=None, create_mask=True, cell_support=False, pre_crop=False,
               output_error=True, check_foreground=True):
    assert im_orig.dtype == np.float32
    crop_props = {}
    crop_props['orig_h'] = im_orig.shape[0]
    crop_props['orig_w'] = im_orig.shape[1]

    if False:
        view = ImageViewer(im_orig)
        view.show()
        sys.exit()

    if pre_crop:
        # rough crop of background to make sure the wafer is near middle of image
        im_uncropped = im_orig.copy()
        b = 20
        im_cols = im_orig.mean(axis=0)
        im_cols -= im_cols.min()
        im_cols /= im_cols.max()
        wafer_cols = np.where(im_cols > 0.2)[0]
        crop_col_start = max(0, wafer_cols.min() - b)
        crop_col_stop = min(len(im_cols), wafer_cols.max() + b)
        im_rows = im_orig.mean(axis=1)
        im_rows -= im_rows.min()
        im_rows /= im_rows.max()
        wafer_rows = np.where(im_rows > 0.2)[0]
        crop_row_start = max(0, wafer_rows.min() - b)
        crop_row_stop = min(len(im_rows), wafer_rows.max() + b)
        im_orig = im_orig[crop_row_start:crop_row_stop, crop_col_start:crop_col_stop]

        if False:
            plt.figure()
            plt.plot(im_rows)
            plt.figure()
            plt.plot(im_cols)
            plt.show()
            sys.exit()

        if False:
            print crop_row_start, crop_row_stop, crop_col_start, crop_col_stop
            view = ImageViewer(im_orig)
            view.show()
            sys.exit()

    if not cell_support:
        # ignore all high intensities as they aren't useful for distinguishing
        #  the wafer from the background
        # NOTE: this isn't necessary for cells

        # for wafers with a very high dynamic range, do a transform to compress it
        f_hist = {}
        smoothed_hist = ip.histogram_percentiles(im_orig, f_hist)
        if f_hist['hist_cov'] > 1:
            if False:
                print f_hist['hist_cov']
                m = int(im_orig.max())
                xs = np.arange(m)
                ImageViewer(im_orig)
                plt.figure()
                plt.plot(xs, smoothed_hist[:m])
                plt.show()
                sys.exit()
            # apply log transform
            im = im_orig.copy()
            pixel_ops.ApplyThresholdLT_F32(im, im, 1, 1)
            im = np.log(im)
        else:
            # clip high intensities
            b = im_orig.shape[0] // 5
            mean1, std1 = cv2.meanStdDev(im_orig[b:-b, b:-b])
            thresh = mean1 - (1.0 * std1)
            im_temp = np.empty_like(im_orig)
            cv2.threshold(im_orig, thresh, thresh, cv2.THRESH_TRUNC, im_temp)
            im = im_temp
    else:
        im = im_orig

    if False:
        print im.min(), im.max(), im.dtype
        view = ImageViewer(im)
        view.show()
        sys.exit()

    # project middle rows/cols of image along axes to get edge profiles
    b = im.shape[0] // 8
    im_height, im_width = im.shape
    h2 = im_height // 2
    w2 = im_width // 2
    buff_size = 50
    row_sum = np.empty(im_height + buff_size * 2, np.float32)
    row_sum[buff_size:-buff_size] = (im[:, w2 - b:w2 + b + 1].sum(axis=1).ravel())
    row_sum[:buff_size] = row_sum[buff_size]
    row_sum[-buff_size:] = row_sum[-buff_size - 1]
    row_sum /= row_sum.max()
    col_sum = np.empty(im_width + buff_size * 2, np.float32)
    col_sum[buff_size:-buff_size] = (im[h2 - b:h2 + b + 1, :].sum(axis=0).ravel())
    col_sum[:buff_size] = col_sum[buff_size]
    col_sum[-buff_size:] = col_sum[-buff_size - 1]
    col_sum /= col_sum.max()

    if cell_support:
        def InterpolateProfile(vals):
            interp = ndimage.gaussian_filter1d(vals, sigma=3, mode="nearest")
            maxes = np.logical_and(interp > np.roll(interp, 1),
                                   interp > np.roll(interp, -1))
            maxes = np.logical_and(maxes, interp > 0.2)
            xs = np.where(maxes)[0]
            if len(xs) < 2: return vals

            f = interp1d(xs, interp[xs], bounds_error=True)
            x1 = xs.min()
            x2 = xs.max()
            interp[x1:x2] = f(np.arange(x1, x2))

            return interp

        row_sum = InterpolateProfile(row_sum)
        col_sum = InterpolateProfile(col_sum)

    if False:
        plt.figure()
        plt.plot(row_sum)
        plt.figure()
        plt.plot(col_sum)
        plt.show()
        sys.exit()

    # check if this image appears to already have been cropped, or if there
    #  is no wafer present
    b = im.shape[0] // 10
    middle_mean = row_sum[b:-b].mean()

    edge_min = min(row_sum[buff_size:buff_size + 20].min(),
                   row_sum[-buff_size - 20:-buff_size].min())  # + 1
    edge_min = max(edge_min, 0.01)
    ratio = middle_mean / edge_min
    if ratio < parameters.EDGE_RATIO:
        if output_error:
            print("\nERROR 1: Cannot find wafer edge. Already been cropped, or wafer missing?")
        raise WaferMissingException
    middle_mean = col_sum[b:-b].mean()
    edge_min = min(col_sum[buff_size:buff_size + 20].min(),
                   col_sum[-buff_size - 20:-buff_size].min())  # + 1
    edge_min = max(edge_min, 0.01)
    ratio = middle_mean / edge_min
    if ratio < parameters.EDGE_RATIO:
        if output_error:
            print ratio, parameters.EDGE_RATIO
            print("\nERROR 2: Cannot find wafer edge. Already been cropped, or wafer missing?")
        raise WaferMissingException

    # differentiate to highlight edges. this is used for guessing width and finding the centre
    if im_orig.shape[0] < 750:
        sigma = 1
    else:
        sigma = 3
    row_sum_smooth = ndimage.gaussian_filter1d(row_sum, sigma, mode="nearest")
    col_sum_smooth = ndimage.gaussian_filter1d(col_sum, sigma, mode="nearest")
    row_diff = row_sum_smooth - np.roll(row_sum_smooth, 1)
    row_diff = np.abs(row_diff)
    row_diff[0] = 0
    row_diff[-1] = 0
    col_diff = col_sum_smooth - np.roll(col_sum_smooth, 1)
    col_diff = np.abs(col_diff)
    col_diff[0] = 0
    col_diff[-1] = 0

    if False:
        plt.figure()
        plt.plot(row_diff)
        plt.figure()
        plt.plot(col_diff)
        plt.show()
        sys.exit()

    if width_prior is None:
        wafer_width = guess_width(im, row_diff, col_diff, buff_size)
    else:
        wafer_width = width_prior

    if False:
        print wafer_width
        plt.figure()
        plt.imshow(im, cmap=cm.gray, interpolation='nearest')
        plt.show()
        sys.exit()

    # estimate the centre of wafer - this will be used as an
    #   initial guess for optimization
    weights = np.zeros(im_height + buff_size * 2, np.int32)
    m = weights.shape[0] // 2
    weights[m - (wafer_width // 2) - 1:m - (wafer_width // 2) + 2] = 1
    weights[m + (wafer_width // 2) - 1:m + (wafer_width // 2) + 2] = 1
    row_cor = ndimage.filters.correlate1d(row_diff, weights, mode="nearest")

    weights = np.zeros(im_width + buff_size * 2, np.int32)
    m = weights.shape[0] // 2
    weights[m - (wafer_width // 2) - 1:m - (wafer_width // 2) + 2] = 1
    weights[m + (wafer_width // 2) - 1:m + (wafer_width // 2) + 2] = 1
    col_cor = ndimage.filters.correlate1d(col_diff, weights, mode="nearest")

    center_x = ndimage.maximum_position(col_cor)[0] - buff_size
    center_y = ndimage.maximum_position(row_cor)[0] - buff_size

    # make sure the guess for the wafer center is somewhere in the middle
    #  of the image
    if (center_x < (im_width / 3) or center_x > ((2 * im_width) / 3) or
                center_y < (im_height / 3) or center_y > ((2 * im_height) / 3)):
        if output_error:
            print("\nERROR: Cannot find wafer middle. Is the wafer broken?")
        raise WaferMiddleException

    if False:
        # print l, r, t, b
        print(center_y, center_x, wafer_width)
        plt.figure()
        plt.imshow(im, cmap=cm.gray, interpolation='nearest')
        plt.figure()
        plt.plot(col_sum)
        plt.figure()
        plt.plot(col_diff)
        plt.figure()
        plt.plot(col_cor)
        plt.show()
        sys.exit()

    # find edges
    if False:
        edge1 = cv2.Sobel(im, cv2.CV_32F, 0, 1)
        edge2 = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    else:
        edge1 = cv2.Sobel(im_orig, cv2.CV_32F, 0, 1)
        edge2 = cv2.Sobel(im_orig, cv2.CV_32F, 1, 0)

    # for bottom of wafer, sobel response is negative, so multiply by -1
    #  to make all our edges of interest positive
    edge1[edge1.shape[0] // 2:, :] *= -1
    edge2[:, edge2.shape[1] // 2:] *= -1
    edges = edge1 + edge2
    edges[:2, :] = 0
    edges[-1, -2:] = 0
    edges[:2, 0] = 0
    edges[-2:, -1] = 0

    # sample x and y coordinates for the edges of a wafer with size wafer_width
    num_samples = 250
    x1 = float(center_x - (wafer_width / 2))
    x2 = float(x1 + wafer_width)
    y1 = float(center_y - (wafer_width / 2))
    y2 = float(y1 + wafer_width)
    ys_l = np.linspace(y1, y2, num_samples).astype(np.float32)
    ys_r = np.linspace(y1, y2, num_samples).astype(np.float32)
    ys_t = np.ones(num_samples, np.float32) * y1
    ys_b = np.ones(num_samples, np.float32) * y2
    xs_l = np.ones(num_samples, np.float32) * x1
    xs_r = np.ones(num_samples, np.float32) * x2
    xs_t = np.linspace(x1, x2, num_samples).astype(np.float32)
    xs_b = np.linspace(x1, x2, num_samples).astype(np.float32)
    ys = np.hstack((ys_l, ys_r, ys_t, ys_b))
    xs = np.hstack((xs_l, xs_r, xs_t, xs_b))

    # we already have estimated for translation and width - find an estimate for rotation
    minR = 0
    minVal = 0
    edges_fine = cv2.GaussianBlur(edges, (0, 0), 2, borderType=cv2.BORDER_CONSTANT)
    for r in np.linspace(-5, 5, 11):
        theta = math.radians(r)
        params = (theta, 0, 0, 1)
        val = pixel_ops.WaferFit(params, center_y, center_x, ys, xs, edges_fine, 0, im_height, im_width)
        if val < minVal:
            minR = theta
            minVal = val

    if False:
        # show the initial guess
        params = (minR, 0, 0, 1)
        pixel_ops.WaferFit(params, center_y, center_x, ys, xs, edges_fine, 1, im_height, im_width)
        view = ImageViewer(edges_fine)
        view.show()
        sys.exit()

    # fine tune the fit using Powell optimisation
    params = (minR, 0, 0, 1)
    args = (center_y, center_x, ys, xs, edges_fine, 0, im_height, im_width)
    op_params = optimize.fmin_powell(pixel_ops.WaferFit, params, args,
                                     ftol=0.1, maxfun=300, disp=False)
    (theta, shift_y, shift_x, scale) = op_params
    rotation = np.rad2deg(theta)

    if width_prior is not None: scale = 1

    if False:
        print op_params
        print pixel_ops.WaferFit(op_params, center_y, center_x, ys, xs, edges_fine, 1, im_height, im_width)
        view = ImageViewer(im_orig)
        ImageViewer(edges_fine)
        view.show()
        # sys.exit()

    # fine tune each edge individually (only for high res images)
    if True:  # im_orig.shape[0] > 700:
        shifts = range(-parameters.WAFER_CROP_TOLERANCE, parameters.WAFER_CROP_TOLERANCE + 1)

        # left
        fits = []
        for shift in shifts:
            fits.append(pixel_ops.WaferFit(op_params, center_y, center_x, ys_l, xs_l + shift, edges_fine, 1, im_height,
                                           im_width))
        x1 += shifts[np.argmin(fits)]

        # right
        fits = []
        for shift in shifts:
            fits.append(pixel_ops.WaferFit(op_params, center_y, center_x, ys_r, xs_r + shift, edges_fine, 1, im_height,
                                           im_width))
        x2 += shifts[np.argmin(fits)]

        # top
        fits = []
        for shift in shifts:
            fits.append(pixel_ops.WaferFit(op_params, center_y, center_x, ys_t + shift, xs_t, edges_fine, 1, im_height,
                                           im_width))
        y1 += shifts[np.argmin(fits)]

        # bottom
        fits = []
        for shift in shifts:
            fits.append(pixel_ops.WaferFit(op_params, center_y, center_x, ys_b + shift, xs_b, edges_fine, 1, im_height,
                                           im_width))
        y2 += shifts[np.argmin(fits)]

    # get rotated coordinates of the wafer corners
    ps = []
    ps.append([y1, x1])  # top left
    ps.append([y1, x2])  # top right
    ps.append([y2, x2])  # bottom right
    ps.append([y2, x1])  # bottom left
    crop_props['corners_floats'] = []
    x_total = 0
    y_total = 0
    for p in ps:
        [y, x] = p
        # coordinates in image
        y_c = (y - center_y) * scale
        x_c = (x - center_x) * scale
        y_r = ((x_c * math.sin(theta)) + (y_c * math.cos(theta)))
        x_r = ((x_c * math.cos(theta)) - (y_c * math.sin(theta)))
        y_r += (center_y + shift_y)
        x_r += (center_x + shift_x)
        crop_props['corners_floats'].append([y_r, x_r])
        p[0] = int(round(y_r))
        p[1] = int(round(x_r))

        x_total += x_r
        y_total += y_r

    # update initial guesses
    wafer_width = float(wafer_width) * scale
    center_y = y_total / 4.0
    center_x = x_total / 4.0

    if pre_crop:
        center_y += crop_row_start
        center_x += crop_col_start
        for p in ps:
            p[0] += crop_row_start
            p[1] += crop_col_start
        im = im_uncropped
        im_height, im_width = im.shape

    crop_props['estimated_width'] = wafer_width
    crop_props['center'] = (center_y, center_x)
    crop_props['estimated_rotation'] = rotation
    crop_props['corners'] = ps

    if create_mask:
        # get coords of bounding box
        t = min(ps[0][0], ps[1][0])
        t2 = max(ps[0][0], ps[1][0])
        b = max(ps[2][0], ps[3][0])
        b2 = min(ps[2][0], ps[3][0])
        l = min(ps[0][1], ps[3][1])
        l2 = max(ps[0][1], ps[3][1])
        r = max(ps[1][1], ps[2][1])
        r2 = min(ps[1][1], ps[2][1])
        if t < 0: t = 0
        if l < 0: l = 0
        if r > im_width: r = im_width
        if b > im_height: b = im_height

        # create foreground/background mask
        edge_zone = np.zeros(im.shape, np.uint8)
        edge_zone[t:t2, l:r] = 1
        edge_zone[b2:b, l:r] = 1
        edge_zone[t:b, l:l2] = 1
        edge_zone[t:b, r2:r] = 1
        ys, xs = np.where(edge_zone == 1)

        mask = np.zeros(im.shape, np.uint8)
        mask[:t, :] = 1
        mask[b:, :] = 1
        mask[:, :l] = 1
        mask[:, r:] = 1

        for i in range(4):
            segment_points = np.array([[ps[i]], [ps[(i + 1) % 4]]])
            segment_vector = segment_points[1] - segment_points[0]

            YX = np.vstack((ys, xs)) - segment_points[0].T
            dp = np.dot(segment_vector, YX)
            dp = np.sign(dp).flat
            mask[ys[dp == -1], xs[dp == -1]] = 1

        crop_props['mask'] = mask

        if False:
            view = ImageViewer(mask)
            ImageViewer(im_uncropped)
            view.show()
            sys.exit()

    # a foreground background comparison to make sure this crop looks good
    if 'mask' in crop_props and check_foreground:
        # view = ImageViewer(im_orig)
        # ImageViewer(crop_props['mask'])
        # view.show()
        if not im_orig.flags['C_CONTIGUOUS']: im_orig = np.ascontiguousarray(im_orig)
        foreground_mean = pixel_ops.MaskMean_F32(im_orig, crop_props['mask'], 0)
        background_mean = pixel_ops.MaskMean_F32(im_orig, crop_props['mask'], 1)
        ratio = foreground_mean / max(1, background_mean)
        # print "\n* ", ratio

        if ratio < parameters.MONO_CROP_SNR:
            print(
            "\nERROR: Wafer foreground not much brighter than image background. Wafer missing or low SnR? %0.02f < %0.02f" % (
            ratio, parameters.MONO_CROP_SNR))
            raise WaferMissingException
        crop_props['foreground_background_ratio'] = ratio

    return crop_props


def crop_wafer_cz(im, force_width=None, create_mask=True, skip_crop=False,
                  cell_support=False, pre_crop=False, output_error=True, check_foreground=True,
                  outermost_peak=False):
    if skip_crop:
        # already cropped
        crop_props = {}
        crop_props['estimated_width'] = im.shape[0]
        crop_props['center'] = (im.shape[0] / 2, im.shape[1] / 2)
        crop_props['corners'] = [[0, 0],
                                 [0, im.shape[1]],
                                 [im.shape[0], im.shape[1]],
                                 [im.shape[0], 0],
                                 ]
        crop_props['corners_floats'] = crop_props['corners']
        crop_props['estimated_rotation'] = 0
        crop_props['mask'] = np.zeros_like(im, np.uint8)
    else:
        # regular cropping to get box
        crop_props = crop_wafer(im.astype(np.float32), force_width, create_mask, cell_support=cell_support,
                                pre_crop=pre_crop, output_error=output_error, check_foreground=check_foreground)

    # determine radius based on the edge profile. this is to handle the
    #  case where the wafer blends into the background
    ps = crop_props['corners']
    y, x = int(round(crop_props['center'][0])), int(round(crop_props['center'][1]))
    profile = np.zeros(im.shape[0], np.float32)
    corner_dist_euclid = 0
    corner_dist_pixels = 0
    h, w = im.shape
    corner_dists = []
    for c in range(4):
        ys, xs = draw.line(y, x, ps[c][0], ps[c][1])
        mask = np.ones(len(xs), dtype=np.bool)
        mask = np.logical_and(mask, ys >= 0)
        mask = np.logical_and(mask, xs >= 0)
        mask = np.logical_and(mask, ys < h)
        mask = np.logical_and(mask, xs < w)
        ys = ys[mask]
        xs = xs[mask]

        corner_dist_euclid += math.sqrt((y - ps[c][0]) ** 2 + (x - ps[c][1]) ** 2)
        corner_dist_pixels += len(ys)
        profile[:len(ys)] += im[ys, xs]
        corner_dists.append(len(ys))

    profile /= 4.0
    corner_dist_euclid /= 4.0
    corner_dist_pixels /= 4.0
    corner_dist = int(round(np.median(corner_dists))) - 1
    profile[corner_dist:] = profile[corner_dist]

    # differentiate profile
    smoothed = ndimage.gaussian_filter1d(profile, 3, mode="nearest")
    df = smoothed - np.roll(smoothed, -1)
    df[0] = 0
    df[-1] = 0
    df /= df.max()

    if outermost_peak:
        # find outermost peak
        for r in range(df.shape[0] - 2, 1, -1):
            if df[r] > df[r - 1] and df[r] > df[r + 1] and df[r] > 0.4:
                break
    else:
        # find highest peak (not too close to center)
        start = int((crop_props['estimated_width'] // 2) / 1.4)
        r = start + np.argmax(df[start:])

    # need to convert from pixel distance to euclidean distance
    radius = (float(r + 1) / corner_dist_pixels) * corner_dist_euclid
    crop_props['radius'] = radius

    if False:
        print crop_props['estimated_width']# // 2
        print r, radius, corner_dist_pixels, corner_dist_euclid

        plt.figure()
        plt.plot(profile)
        plt.plot(smoothed)
        plt.plot(r+1, smoothed[r+1], 'o')
        plt.figure()
        plt.plot(r + 1, df[r + 1], 'o')
        plt.plot(df)
        plt.show()
        # sys.exit()

    if create_mask:
        # set anything outside of radius to background
        Y, X = np.mgrid[:im.shape[0], :im.shape[1]]
        Y -= crop_props['center'][0]
        X -= crop_props['center'][1]
        dist = np.sqrt((X ** 2) + (Y ** 2))
        crop_props['mask'][dist > radius] = 1

    return crop_props


def crop_cell_rd(im_orig, crop_busbars=True, crop_mode=1):
    crop_props = {}

    # robust scaling (use 5th and 95th percentiles)
    im = cv2.medianBlur(im_orig, ksize=3)
    hist = np.zeros(2 ** 16, np.int32)
    pixel_ops.FastHistogram(im.astype(np.float32), hist)
    cdf = np.cumsum(hist) / float(im.shape[0] * im.shape[1])
    pixel01 = 0.05
    hist01 = np.where(cdf > pixel01)[0][0]
    pixel95 = 0.95
    hist95 = np.where(cdf > pixel95)[0][0]
    im = (im.astype(np.float32) - hist01) / float(hist95 - hist01)
    pixel_ops.ApplyThresholdLT_F32(im, im, 0, 0)
    pixel_ops.ApplyThresholdGT_F32(im, im, 1, 1)

    # make sure there is a wafer
    s = int(0.25 * im_orig.shape[0])
    middle_mean = im[s:-s:2, s:-s:2].mean()
    edge_mean = (im[:10, :].mean() + im[-10:, :].mean() +
                 im[:, :10].mean() + im[:, -10:].mean()) / 4.0
    ratio = middle_mean / max(0.1, edge_mean)
    if ratio < 2: raise WaferMissingException

    # ROI crop
    col_mean = im.mean(axis=0)
    crop_l = max(0, np.where(col_mean > 0.02)[0][0] - 10)
    crop_r = min(im.shape[1], np.where(col_mean > 0.02)[0][-1] + 10)
    row_mean = im.mean(axis=1)
    crop_t = max(0, np.where(row_mean > 0.02)[0][0] - 10)
    crop_b = min(im.shape[0], np.where(row_mean > 0.02)[0][-1] + 10)
    im = im[crop_t:crop_b, crop_l:crop_r]
    crop_props['corners_roi'] = [crop_l, crop_t, crop_r, crop_b]

    if False:
        print crop_t, crop_b
        view = ImageViewer(im_orig)
        ImageViewer(im)
        view.show()

    if crop_mode == 0:
        mask = np.ones_like(im_orig, dtype=np.uint8)
        mask[crop_t:crop_b, crop_l:crop_r] = 0
        crop_props['mask'] = mask
        return crop_props

    # estimate edges
    col_mean = im.mean(axis=0)
    row_mean = im.mean(axis=1)

    def profile_fit(args, profile):
        start, stop = args
        if stop - start < 10: return 100
        if start <= 0: return 100
        if stop >= len(profile): return 100
        a = int(max(1, start))
        b = int(min(len(profile) - 2, stop))
        background_l = profile[:a].mean()
        background_r = profile[b:].mean()
        # print a, b, background_l, background_r
        score = (background_l + background_r) - profile[a:b].mean()
        return score

    w = len(col_mean)
    h = len(row_mean)
    (edge_l, edge_r) = optimize.fmin_powell(profile_fit, (1, w - 1), args=(col_mean,), disp=False, xtol=0.1, ftol=0.001)
    (edge_t, edge_b) = optimize.fmin_powell(profile_fit, (1, h - 1), args=(row_mean,), disp=False, xtol=0.1, ftol=0.001)
    edge_l = int(round(edge_l))
    edge_r = int(round(edge_r))
    edge_t = int(round(edge_t))
    edge_b = int(round(edge_b))

    if False:
        print edge_l, edge_r, edge_t, edge_b
        plt.figure()
        plt.plot(col_mean)
        plt.axvline(edge_l, color="r")
        plt.axvline(edge_r, color="r")
        plt.figure()
        plt.axvline(edge_t, color="r")
        plt.axvline(edge_b, color="r")
        plt.plot(row_mean)
        plt.show()

    if crop_mode == 1:
        # RECTANGLE CROP
        # fine tune edges and get rotation
        center_y = im.shape[0] // 2
        center_x = im.shape[1] // 2
        mask = np.empty_like(im, dtype=np.float32)
        rotated = np.empty_like(im, dtype=np.float32)
        im = np.ascontiguousarray(im)

        def goodness_of_fit(args):
            t, b, l, r, rotation = args
            pixel_ops.InitRectangle(mask, t, b, l, r)
            rot_mat = cv2.getRotationMatrix2D((center_y, center_x), -rotation, 1.0)
            cv2.warpAffine(mask, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, dst=rotated)
            if False:
                dist = im[rotated < 0.5].mean() - im[rotated > 0.5].mean()
            else:
                foreground_mean, background_mean = pixel_ops.MaskAvgDiff(im, rotated)
                dist = background_mean - foreground_mean
            return dist

        t1 = timeit.default_timer()
        t, b, l, r, rotation = optimize.fmin_powell(goodness_of_fit, (edge_t, edge_b, edge_l, edge_r, 0), disp=False,
                                                    xtol=1.0, ftol=0.05)
        t = max(0, int(round(t)))
        b = min(im.shape[0], int(round(b)))
        l = max(0, int(round(l)))
        r = min(im.shape[1], int(round(r)))
        t2 = timeit.default_timer()

        crop_props['estimated_width'] = b - t
        crop_props['estimated_height'] = l - r
        crop_props['estimated_rotation'] = rotation

        if False:
            print t2 - t1
            pixel_ops.InitRectangle(mask, t, b, l, r)
            rot_mat = cv2.getRotationMatrix2D((center_y, center_x), -rotation, 1.0)
            cv2.warpAffine(mask, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST,
                           borderMode=cv2.BORDER_CONSTANT, dst=rotated)
            view = ImageViewer(im)
            view = ImageViewer(rotated)
            view.show()
            sys.exit()

        # unrotate cell
        rot_mat = cv2.getRotationMatrix2D((center_y, center_x), rotation, 1.0)
        cell = cv2.warpAffine(im, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT)[t:b, l:r]
        mask = np.ones_like(cell, dtype=np.uint8)

        if crop_busbars:
            # remove busbars
            cell_cols = np.apply_along_axis(stats.scoreatpercentile, 0, cell, 90)
            cell_rows = np.apply_along_axis(stats.scoreatpercentile, 1, cell, 90)
            mask[:, cell_cols < 0.6] = 0
            mask[cell_rows < 0.6, :] = 0
            if False:
                view = ImageViewer(im)
                view = ImageViewer(cell)
                view = ImageViewer(mask)
                plt.figure()
                plt.plot(cell_cols)
                plt.ylim(0, 1)
                plt.figure()
                plt.plot(cell_rows)
                plt.ylim(0, 1)
                view.show()
                sys.exit()

        # transform mask back to original dimensions
        mask = np.pad(mask, ((t, im.shape[0] - t - mask.shape[0]), (l, im.shape[1] - l - mask.shape[1])),
                      mode="constant", constant_values=0)
        assert im.shape == mask.shape
        rot_mat = cv2.getRotationMatrix2D((center_y, center_x), -rotation, 1.0)
        mask = cv2.warpAffine(mask, rot_mat, (im.shape[1], im.shape[0]), flags=cv2.INTER_NEAREST,
                              borderMode=cv2.BORDER_CONSTANT)
        mask = np.pad(mask, ((crop_t, im_orig.shape[0] - mask.shape[0] - crop_t),
                             (crop_l, im_orig.shape[1] - mask.shape[1] - crop_l)), mode="constant", constant_values=0)
        assert im_orig.shape == mask.shape
        crop_props['mask'] = mask
    elif crop_mode == 2:
        # IRREGULAR CROP
        # - crop based on thresholding
        foreground_mask = np.zeros_like(im_orig, np.uint8)
        foreground_mask[crop_t + edge_t:crop_t + edge_b,
                        crop_l + edge_l:crop_l + edge_r] = 1
        foreground_pixels = im_orig[foreground_mask == 1].astype(np.float32)
        foreground_hist = np.zeros(2 ** 16, np.int32)
        pixel_ops.FastHistogram1D(foreground_pixels, foreground_hist)
        background_pixels = im_orig[foreground_mask == 0].astype(np.float32)
        background_hist = np.zeros(2 ** 16, np.int32)
        pixel_ops.FastHistogram1D(background_pixels, background_hist)
        foreground_hist = foreground_hist.astype(np.float32)
        foreground_hist /= foreground_hist.max()
        background_hist = background_hist.astype(np.float32)
        background_hist /= background_hist.max()
        threshold = (np.argmax(foreground_hist) - np.argmax(background_hist)) / 2.0
        foreground_mask = (im_orig > threshold).astype(np.uint8)

        if False:
            plt.figure()
            plt.plot(ndimage.gaussian_filter1d(foreground_hist.astype(np.float32), sigma=50))
            plt.plot(ndimage.gaussian_filter1d(background_hist.astype(np.float32), sigma=50))

            view = ImageViewer(im_orig)
            view = ImageViewer(im_orig > threshold)
            view.show()

        # find convex hull of all large foreground objects
        ccs, count = ip.connected_components(foreground_mask)
        cc_sizes = np.zeros(count + 1, np.int32)
        pixel_ops.CCSizes(ccs, cc_sizes)
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        mask = np.zeros_like(im_orig, np.uint8)
        for cc in range(count + 1):
            if cc_sizes[cc] < 100: continue
            cc_mask = (ccs == cc).astype(np.uint8)
            outline = cc_mask - cv2.erode(cc_mask, kernel=se, iterations=1)
            cv_hull = convex_hull_image(outline)
            mask[cv_hull] = 1

        if False:
            view = ImageViewer(im_orig)
            view = ImageViewer(mask)
            view.show()

        crop_props['mask'] = mask
    else:
        print("UNKNOWN CROP MODE")

    return crop_props


def crop_wafer_rd(im_orig, border_erode=parameters.BORDER_ERODE, rounded_corners=True):
    h, w = im_orig.shape
    crop_props = {}

    # robust scaling (use 1 and 95 percentiles)
    im = np.sqrt(cv2.medianBlur(im_orig, ksize=3))
    hist = np.zeros(2 ** 16, np.int32)
    pixel_ops.FastHistogram(im, hist)
    cdf = np.cumsum(hist) / float(im_orig.shape[0] * im_orig.shape[1])
    pixel01 = 0.01
    hist01 = np.where(cdf > pixel01)[0][0]
    pixel95 = 0.95
    hist95 = np.where(cdf > pixel95)[0][0]
    im = (im.astype(np.float32) - hist01) / float(hist95 - hist01)
    pixel_ops.ApplyThresholdLT_F32(im, im, 0, 0)
    pixel_ops.ApplyThresholdGT_F32(im, im, 1, 1)

    # make sure there is a wafer
    s = int(0.25 * w)
    middle_mean = im[s:-s:2, s:-s:2].mean()
    edge_mean = (im[:10, :].mean() + im[-10:, :].mean() +
                 im[:, :10].mean() + im[:, -10:].mean()) / 4.0
    ratio = middle_mean / max(0.1, edge_mean)
    if ratio < 2: raise WaferMissingException

    if False:
        view = ImageViewer(im_orig)
        view = ImageViewer(im)
        view.show()
        sys.exit()

    # enhance edges
    f = np.ones((5, 40), np.float32)
    f[1:-1, :] = 0
    f[0, :] *= -1
    edge1 = ndimage.convolve(im, weights=f, mode="constant", cval=0)
    edge1[:h // 2, :] *= -1
    edge1[:2, :] = 0
    edge1[-2:, :] = 0
    pixel_ops.ApplyThresholdLT_F32(edge1, edge1, 0, 0)
    edge2 = ndimage.convolve(im, weights=f.T, mode="constant", cval=0)
    edge2[:, :w // 2] *= -1
    edge2[:, :2] = 0
    edge2[:, -2:] = 0
    pixel_ops.ApplyThresholdLT_F32(edge2, edge2, 0, 0)
    edges = np.sqrt((edge1 ** 2) + (edge2 ** 2))

    # correct rotation
    rot_mat, rot_mat_I, rotated_edges, rotation = approx_rotation(edges)
    rotated_im = cv2.warpAffine(im, rot_mat, (w, h),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
    crop_props['estimated_rotation'] = rotation

    if False:
        view = ImageViewer(im)
        view = ImageViewer(edges)
        view = ImageViewer(rotated_edges)
        view.show()
        sys.exit()

    # find edges
    col_mean = rotated_edges.mean(axis=0)
    row_mean = rotated_edges.mean(axis=1)
    mid = h // 2
    y1 = np.argmax(row_mean[:mid])
    y2 = mid + np.argmax(row_mean[mid:])
    mid = w // 2
    x1 = np.argmax(col_mean[:mid])
    x2 = mid + np.argmax(col_mean[mid:])

    crop_props['estimated_width'] = x2 - x1
    crop_props['estimated_height'] = y2 - y1
    crop_props['center'] = ((y2 - y1) // 2, (x2 - x1) // 2)
    crop_props['radius'] = 0

    if False:
        print x2 - x1
        print y2 - y1
        rotated_im[y1, :] = rotated_im.max()
        rotated_im[y2, :] = rotated_im.max()
        rotated_im[:, x1] = rotated_im.max()
        rotated_im[:, x2] = rotated_im.max()
        view = ImageViewer(rotated_im)
        view.show()
        sys.exit()

    # create foreground mask
    foreground = np.ones_like(im_orig, dtype=np.uint8)
    foreground[y1 - 1:y2 + 1, x1 - 1:x2 + 1] = 0

    if rounded_corners:
        # crop wafer
        wafer = cv2.GaussianBlur(rotated_im[y1 - 1:y2 + 1, x1 - 1:x2 + 1], (0, 0), 1, borderType=cv2.BORDER_CONSTANT)

        # only interested in darkest intensities, as this is where the
        #  transition from background to foreground occurs
        t = stats.scoreatpercentile(wafer[::2, ::2].flat, per=20)
        pixel_ops.ApplyThresholdGT_F32(wafer, wafer, t, t)

        if False:
            view = ImageViewer(wafer)
            view.show()
            sys.exit()

        h, w = wafer.shape
        corners = np.ones_like(wafer, np.uint8)
        ys, xs = draw.polygon(np.array([0, h // 2, h - 1, h // 2]),
                              np.array([w // 2, w - 1, w // 2, 0]))
        corners[:, :w // 2] += 1
        corners[:h // 2, :] += 2
        corners[ys, xs] = 0

        # find edge candidates
        filtered = np.zeros_like(wafer)
        edges = np.zeros_like(wafer, np.uint8)
        pixel_ops.EdgeCandidates(wafer, corners, filtered, edges)
        ip.remove_small_ccs(edges, 30)

        # remove small ones, and get coordinates of remaining ones
        ccs, num_ccs = ip.connected_components(edges)
        cc_sizes = np.zeros(num_ccs + 1, np.int32)
        pixel_ops.CCSizes(ccs, cc_sizes)
        coords = np.zeros((num_ccs + 1, cc_sizes.max(), 2), np.int32)
        counts = np.zeros(num_ccs + 1, np.int32)
        pixel_ops.GetCCCoords(ccs, coords, counts)

        # modified hough transform to find most likely center
        acc = np.zeros((h * 3, w * 3), np.int32)
        pixel_ops.AccumulateCircleCenter(coords, counts, acc, h, w)

        if False:
            view = ImageViewer(wafer)
            view = ImageViewer(corners)
            view = ImageViewer(filtered)
            view = ImageViewer(edges)
            view = ImageViewer(acc)
            view.show()
            sys.exit()

        # we now have a guess for the center - figure out the radius
        profile = np.zeros(h + w, np.float32)
        counts = np.ones(h + w, np.float32)
        y, x = ndimage.maximum_position(acc)
        if acc[y, x] > 100:
            y -= h
            x -= w
            pixel_ops.RadialAverage(filtered, profile, counts, h + w, y, x)
            profile[counts < 20] = 0
            profile /= counts
            radius = np.argmax(profile)
            crop_props['radius'] = radius

            # set anything outside of radius to background
            ys, xs = np.mgrid[:h, :w]
            ds = np.sqrt((ys - y) ** 2 + (xs - x) ** 2)
            mask = ds > radius
            ys += y1 - 1
            xs += x1 - 1
            foreground[ys[mask], xs[mask]] = 1

            if False:
                plt.figure()
                plt.plot(profile)
                plt.show()
                sys.exit()

    if False:
        view = ImageViewer(rotated_im)
        view = ImageViewer(foreground)
        view.show()
        sys.exit()

    # undo rotation
    foreground = cv2.warpAffine(foreground, rot_mat_I, (im_orig.shape[1], im_orig.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)

    if False:
        view = ImageViewer(im_orig)
        view = ImageViewer(foreground)
        view.show()
        sys.exit()

    # border erosion
    se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    if border_erode > 0:
        foreground = cv2.erode(foreground, kernel=se, iterations=border_erode)

    # outer BB
    # borders = ndimage.binary_dilation(foreground - cv2.erode(foreground, kernel=se), se)
    borders = cv2.dilate(foreground - cv2.erode(foreground, kernel=se), se)
    ys, xs = np.where(borders == 1)
    out_y1 = ys.min()
    out_y2 = ys.max()
    out_x1 = xs.min()
    out_x2 = xs.max()

    # inner BB
    middle_y = (out_y1 + out_y2) // 2
    middle_x = (out_x1 + out_x2) // 2

    corners = pixel_ops.GetDiags(foreground, middle_y, middle_x)

    in_y1 = max(corners[0][0], corners[1][0])
    in_y2 = min(corners[2][0], corners[3][0])
    in_x1 = max(corners[0][1], corners[3][1])
    in_x2 = min(corners[1][1], corners[2][1])

    crop_props['mask'] = foreground
    crop_props['borders'] = borders
    crop_props['corners_inner'] = [(in_y1, in_x1), (in_y1, in_x2),
                                   (in_y2, in_x1), (in_y2, in_x2)]
    crop_props['corners_outer'] = [(out_y1, out_x1), (out_y1, out_x2),
                                   (out_y2, out_x1), (out_y2, out_x2)]

    if __name__ == '__main__':
        # draw results
        results = ip.overlay_mask(im_orig, mask=borders)

        # outer BB
        results[out_y1:out_y2, out_x1:out_x1 + 2, :] = (255, 0, 0)
        results[out_y1:out_y2, out_x2 - 2:out_x2:, :] = (255, 0, 0)
        results[out_y1:out_y1 + 2, out_x1:out_x2, :] = (255, 0, 0)
        results[out_y2 - 2:out_y2, out_x1:out_x2, :] = (255, 0, 0)

        # inner BB
        results[in_y1:in_y2, in_x1:in_x1 + 2, :] = (0, 0, 255)
        results[in_y1:in_y2, in_x2 - 2:in_x2:, :] = (0, 0, 255)
        results[in_y1:in_y1 + 2, in_x1:in_x2, :] = (0, 0, 255)
        results[in_y2 - 2:in_y2, in_x1:in_x2, :] = (0, 0, 255)

        view = ImageViewer(results)
        view.show()
        sys.exit()

    return crop_props


# @profile
def crop_cell(rotated, orig, features, width=None, already_cropped=False):
    # set background to 0
    norm = rotated.copy()

    # rough normalization (will refine after cropping)
    if norm.mean() > 10:
        f = {}
        ip.histogram_percentiles(norm, f)
        norm /= f['hist_median']

    if False:
        view = ImageViewer(norm)
        view.show()

    if already_cropped:
        top, bottom = 0, norm.shape[0]
        left, right = 0, norm.shape[1]
    elif width is None:
        # due to belts, only use the middle of the image
        h2 = norm.shape[0] // 2
        hd = norm.shape[0] // 10
        w2 = norm.shape[1] // 2
        wd = norm.shape[1] // 10

        foreground_thresh = parameters.CELL_BACKGROUND_THRESH

        # try to find outermost edges
        window = norm[:, w2 - wd:w2 + wd]
        window = cv2.medianBlur(window, ksize=5)
        rows = np.median(window, axis=1)
        rows = ndimage.gaussian_filter1d(rows, sigma=1)
        diff_r = np.roll(rows, -1) - np.roll(rows, 1)
        diff_r[len(diff_r) // 2:] *= -1
        diff_r[[0, 1, -2, -1]] = 0
        diff_r /= diff_r.max()
        foreground_mask = (rows > foreground_thresh)
        peaks_r = np.where(((diff_r > np.roll(diff_r, 1)) & (diff_r > np.roll(diff_r, -1)) &
                            (rows > 0.1) & (diff_r > parameters.CELL_DIFF_THRESH)) | foreground_mask)[0]

        if len(peaks_r) < 2:
            print("\nERROR: Cannot find top/bottom cell edges. Already been cropped, or wafer missing?")
            raise CellMissingException
        top, bottom = peaks_r[[0, -1]]
        top = pixel_ops.HillClimb1D(diff_r, top)
        bottom = pixel_ops.HillClimb1D(diff_r, bottom)

        window = norm[h2 - hd:h2 + hd, :]
        window = cv2.medianBlur(window, ksize=5)
        cols = np.median(window, axis=0)
        cols = ndimage.gaussian_filter1d(cols, sigma=1)
        diff_c = np.roll(cols, -1) - np.roll(cols, 1)
        diff_c[len(diff_c) // 2:] *= -1
        diff_c[[0, 1, -2, -1]] = 0
        diff_c /= diff_c.max()
        peaks_c = np.where(((diff_c > np.roll(diff_c, 1)) & (diff_c > np.roll(diff_c, -1)) &
                            (cols > 0.1) & (diff_c > parameters.CELL_DIFF_THRESH)) |
                           (norm.mean(axis=0) > foreground_thresh))[0]
        if len(peaks_c) < 2:
            print("\nERROR: Cannot find left/right cell edge. Already been cropped, or wafer missing?")
            raise CellMissingException
        left, right = peaks_c[[0, -1]]
        left = pixel_ops.HillClimb1D(diff_c, left)
        right = pixel_ops.HillClimb1D(diff_c, right)

        if False:
            print features['_alg_mode']
            plt.figure()
            plt.plot(diff_c)
            plt.plot([left, right], diff_c[[left, right]], 'o')
            plt.title("Left & right")
            plt.figure()
            plt.plot(diff_r)
            plt.plot([top, bottom], diff_r[[top, bottom]], 'o')
            plt.title("Top & bottom")
            plt.figure()
            plt.plot(rows)
            plt.vlines([top, bottom], ymin=0, ymax=rows.max())
            # profile = cols#norm.mean(axis=0)
            plt.figure()
            plt.plot(cols)
            # plt.plot(norm[h2-hd:h2+hd, :].mean(axis=0))
            plt.vlines([left, right], ymin=0, ymax=cols.max())
            ImageViewer(norm)
            plt.show()

        # make sure width roughly equals height
        width, height = right - left, bottom - top
        aspect_ratio = max(width, height) / max(1.0, float(min(width, height)))
        if width == 0 or height == 0 or aspect_ratio > 1.1:
            print("\nERROR: Cropping error. Cell does not approximate a square. Ratio: %0.02f" % aspect_ratio)
            raise CellMissingException

        # make sure cell is brighter than background
        mask = np.zeros_like(norm, np.bool)
        mask[top:bottom, left:right] = True
        foreground = np.median(norm[mask])
        background = max(0.01, np.median(norm[~mask]))

        ratio = foreground / background
        if ratio < parameters.MONO_CROP_SNR:
            print("\nERROR: Cropping error. Cell not brighter than background.")
            raise CellMissingException

        left += parameters.CELL_BORDER_ERODE
        top += parameters.CELL_BORDER_ERODE
        right -= parameters.CELL_BORDER_ERODE
        bottom -= parameters.CELL_BORDER_ERODE
    else:
        cols = ndimage.gaussian_filter1d(norm.mean(axis=0), sigma=1)
        vals = [cols[i] + cols[i + width] for i in range(len(cols) - width)]
        left = np.argmin(vals)
        right = left + width

        rows = ndimage.gaussian_filter1d(norm.mean(axis=1), sigma=1)
        vals = [rows[i] + rows[i + width] for i in range(len(rows) - width)]
        top = np.argmin(vals)
        bottom = top + width

        if False:
            print left, right
            plt.figure()
            plt.plot(cols)
            plt.vlines([left, right], ymin=0, ymax=cols.max())
            plt.show()

    features['crop_left'] = left
    features['crop_right'] = right
    features['crop_top'] = top
    features['crop_bottom'] = bottom
    cropped = np.ascontiguousarray(norm[top:bottom, left:right])

    if False:
        print right - left
        print bottom - top
        view = ImageViewer(orig)
        view = ImageViewer(cropped)
        view.show()

    # get approx radius
    h, w = cropped.shape
    diag = math.sqrt((h / 2.0) ** 2 + (w / 2.0) ** 2)
    profile = np.zeros(h, np.float32)
    counts = np.zeros(h, np.float32)
    pixel_ops.RadialAverage(cropped, profile, counts, h, h // 2, w // 2)
    counts[counts < 1] = 1
    profile /= counts
    profile /= profile.max()
    profile = profile[:int(diag) - 1]

    # make force radius to be 80% of distance to corner
    p80 = int(0.8 * diag)
    profile[:p80] = profile[p80]

    diff = np.abs(np.roll(profile, -1) - np.roll(profile, 1))
    diff[[0, -1]] = 0
    peaks = np.where((diff > np.roll(diff, 1)) &
                     (diff > np.roll(diff, -1)) &
                     (diff > parameters.CELL_EDGE_THRESH))[0]
    if len(peaks) > 0:
        radius_estimate = peaks[-1]
    else:
        radius_estimate = int(diag)

    if False:
        r = int(round(radius_estimate))
        print diag, r
        rr, cc = draw.circle_perimeter(h // 2, w // 2, r)
        mask = (rr >= 0) & (rr < h) & (cc >= 0) & (cc < w)
        rr = rr[mask]
        cc = cc[mask]
        cropped[rr, cc] = cropped.max()
        plt.figure()
        plt.plot(profile)
        plt.plot(diff)
        plt.vlines(radius_estimate, 0, 1)
        plt.vlines(radius_estimate, 0, 1)
        view = ImageViewer(cropped)
        view.show()
        sys.exit()

    # if estimate is more than 95% of the way to the corner,
    #  assume this is a square
    percent_to_edge = 100.0 * (radius_estimate / diag)
    if percent_to_edge < 60:
        print("\nERROR: Cropping error. Could not estimate radius.")
        raise CellMissingException
    features['_cell_diag'] = diag

    features['wafer_radius'] = radius_estimate
    features['wafer_middle_x'] = w // 2
    features['wafer_middle_y'] = h // 2

    crop_corners(cropped, norm, features)

    return np.ascontiguousarray(rotated[top:bottom, left:right])


def crop_corners(cropped, uncropped, features):
    h, w = cropped.shape
    top, bottom = features['crop_top'], features['crop_bottom']
    left, right = features['crop_left'], features['crop_right']

    # get original coordinates of the wafer corners
    ps = []
    ps.append([top, left])  # top left
    ps.append([top, right])  # top right
    ps.append([bottom, right])  # bottom right
    ps.append([bottom, left])  # bottom left
    ps.append([h // 2 + top, w // 2 + left])  # middle

    if 'cell_rotated' in features and not features['cell_rotated']:
        center_y, center_x = uncropped.shape[0] // 2, uncropped.shape[1] // 2
    else:
        center_y, center_x = uncropped.shape[0] // 2, uncropped.shape[0] // 2

    theta = math.radians(features['crop_rotation'])
    shift_y, shift_x = 0, 0
    for p in ps:
        [y, x] = p
        # coordinates in image
        y_c = (y - center_y)
        x_c = (x - center_x)
        y_r = ((x_c * math.sin(theta)) + (y_c * math.cos(theta)))
        x_r = ((x_c * math.cos(theta)) - (y_c * math.sin(theta)))
        y_r += (center_y + shift_y)
        x_r += (center_x + shift_x)
        p[0] = int(round(y_r))
        p[1] = int(round(x_r))
    features['_wafer_middle_orig'] = ps.pop()
    if 'cell_rotated' in features and features['cell_rotated']:
        ps = [ps[3]] + ps[:3]

    features['corners'] = ps
    features['corner_tl_x'] = ps[0][1]
    features['corner_tl_y'] = ps[0][0]
    features['corner_tr_x'] = ps[1][1]
    features['corner_tr_y'] = ps[1][0]
    features['corner_br_x'] = ps[2][1]
    features['corner_br_y'] = ps[2][0]
    features['corner_bl_x'] = ps[3][1]
    features['corner_bl_y'] = ps[3][0]

    if False:
        pprint(features)
        # view = ImageViewer(cropped)
        uncropped[ps[0][0], ps[0][1]] = uncropped.max()
        uncropped[ps[1][0], ps[1][1]] = uncropped.max()
        uncropped[ps[2][0], ps[2][1]] = uncropped.max()
        uncropped[ps[3][0], ps[3][1]] = uncropped.max()
        view = ImageViewer(uncropped)
        view.show()
        sys.exit()


def correct_stripe_rotation(im, features, already_cropped=False):

    if already_cropped:
        rotated = im
        features['crop_rotation'] = 0
        print("rotation skipped")
    else:
        # rotation correction
        f = {}
        ip.histogram_percentiles(im, f)
        norm = im / f['hist_percentile_99']
        bw = (norm > 0.2).astype(np.uint8)

        if False:
            view = ImageViewer(norm)
            ImageViewer(bw)
            view.show()
            return

        maskPoints = cv2.findNonZero(bw)
        rect = cv2.minAreaRect(maskPoints)
        rotation = rect[2]
        if rotation < -80:
            rotation += 90
        if rotation > 90:
            rotation -= 90
        M = cv2.getRotationMatrix2D(rect[0], rotation, 1.0)
        rotated = cv2.warpAffine(im, M, (norm.shape[1], norm.shape[0]))
        features['crop_rotation'] = rotation

        if False:
            view = ImageViewer(norm)
            ImageViewer(rotated)
            view.show()
            return

    return rotated


def crop_stripe(orig, rotated, features, already_cropped=False):

    if already_cropped:
        top, bottom = 0, rotated.shape[0]
        left, right = 0, rotated.shape[1]
        print("cropping skipped")
    else:
        # crop
        cols = ndimage.gaussian_filter1d(rotated.mean(axis=0), 5)
        cols -= cols.min()
        cols /= cols.max()
        foreground_cols = np.where(cols > 0.2)[0]
        rows = ndimage.gaussian_filter1d(rotated.mean(axis=1), 5)
        rows -= rows.min()
        rows /= rows.max()
        foreground_rows = np.where(rows > 0.2)[0]

        top, bottom = foreground_rows[0], foreground_rows[-1]
        left, right = foreground_cols[0], foreground_cols[-1]

    features['crop_top'] = top
    features['crop_bottom'] = bottom
    features['crop_left'] = left
    features['crop_right'] = right

    strip_cropped = rotated[top:bottom, left:right]
    strip_cropped = np.ascontiguousarray(strip_cropped)

    if False:
        print features['crop_top']
        view = ImageViewer(orig)
        ImageViewer(strip_cropped)
        view.show()

    crop_corners(strip_cropped, orig, features)

    return strip_cropped


def correct_cell_rotation(im, features, module_cell=False, already_cropped=False):
    h, w = im.shape

    if not already_cropped:
        if not module_cell:
            # estimate background intensity
            vals = np.r_[im[:, :5].flat, im[:, -5:].flat, im[:5, :].flat, im[-5:, :].flat]
            background = np.median(vals)
        else:
            background = 0

        if im[::4, ::4].mean() > 1:
            # rough normalization (will refine after cropping)
            norm = im - background
            f = {}
            ip.histogram_percentiles(norm, f)
            norm /= f['hist_percentile_90']
            pixel_ops.ApplyThresholdLT_F32(norm, norm, 0, 0)
        else:
            # input has already been normalized
            norm = im

        if False:
            view = ImageViewer(im)
            ImageViewer(norm)
            view.show()
            sys.exit()

        # find middle
        im_mean = norm.mean()
        im_foreground = norm.copy()
        im_foreground[im_foreground > im_mean] = im_mean
        rows = im_foreground.mean(axis=0)
        cdf = np.cumsum(rows)
        cdf /= cdf[-1]
        middle_x = np.where(cdf > 0.5)[0][0]
        cols = im_foreground.mean(axis=1)
        cdf = np.cumsum(cols)
        cdf /= cdf[-1]
        middle_y = np.where(cdf > 0.5)[0][0]

        ##################
        # Rough estimate #
        ##################
        # estimate rotation by looking at top edge
        # first find a foreground threshold
        left, right = middle_x - h // 5, middle_x + h // 5
        # hill climb left & right to local maxes (so not using BB col)
        cols = ndimage.gaussian_filter1d(norm.mean(axis=0), sigma=2)
        left = ip.closest_local_max(cols, left)
        right = ip.closest_local_max(cols, right)

        if False:
            plt.figure()
            plt.plot(cols)
            plt.vlines([left, right], 0, 1)
            plt.figure()
            plt.imshow(norm, interpolation='nearest', cmap=cm.gray)
            plt.vlines([left, right], 0, norm.shape[0], 'r')
            plt.show()
            sys.exit()

        left_diff = ndimage.gaussian_filter1d(norm[:, left] - np.roll(norm[:, left], 3), sigma=2)
        zeros = np.zeros(50, np.float32)
        left_diff = np.r_[zeros, left_diff, zeros]
        right_diff = ndimage.gaussian_filter1d(norm[:, right] - np.roll(norm[:, right], 3), sigma=2)
        right_diff = np.r_[zeros, right_diff, zeros]
        shifts = np.arange(-50, 51)
        scores = np.array([(-1 * np.abs(right_diff - np.roll(left_diff, s))).mean() for s in shifts])
        scores -= scores.min()
        top_left = 10
        lms = ((scores > np.roll(scores, 1)) & (scores > np.roll(scores, -1)))
        lms[[0, -1]] = False
        local_maxs = np.where(lms)[0]

        if len(local_maxs) == 0:
            dy = 0
        elif len(local_maxs) == 1:
            dy = shifts[local_maxs[0]]
        elif len(local_maxs) >= 2:
            # find difference
            ordered = np.argsort(scores[local_maxs])
            sorted_peaks = scores[local_maxs][ordered]
            p1, p2 = sorted_peaks[-1], sorted_peaks[-2]
            ratio = p1 / p2
            if ratio > 1.3:
                # one is clearly a better match, so select it
                dy = shifts[local_maxs[scores[local_maxs].argmax()]]
            elif ratio < 1.01:
                # almost equal, so pick closest to middle
                dy = shifts[local_maxs[np.argsort(np.abs(local_maxs - (len(shifts) // 2)))[0]]]
            else:
                # no clear winner, so pick highest of 2 local maxes closest to middle
                closest_peaks = local_maxs[np.argsort(np.abs(local_maxs - (len(shifts) // 2)))[:2]]
                p1, p2 = scores[closest_peaks]
                # ratio = max()
                # print p1, p2, ratio
                if p1 > p2:
                    dy = shifts[closest_peaks[0]]
                else:
                    dy = shifts[closest_peaks[1]]

        top_right = top_left + dy
        dx = right - left
        if False:
            print dy
            plt.figure()
            plt.plot([left, right], [top_left, top_right])
            plt.imshow(im, interpolation='nearest', cmap=cm.gray)
            plt.figure()
            plt.plot(left_diff)
            plt.plot(right_diff)
            plt.figure()
            plt.plot(shifts, scores)
            i = np.where(shifts == dy)[0][0]
            plt.plot(dy, scores[i], 'o')
            plt.show()
        rotation_rough = math.degrees(math.atan2(dy, dx))

        #####################
        # Make BBs vertical #
        #####################
        s5 = int(round(h * 0.4))
        y1, y2 = max(0, middle_y - s5), min(middle_y + s5, h)
        x1, x2 = max(0, middle_x - s5), min(middle_x + s5, w)
        middle = norm[y1:y2, x1:x2]
        rotated_middle = np.empty_like(middle)
        middle_h, middle_w = middle.shape
        rot_mat = cv2.getRotationMatrix2D((middle_w // 2, middle_h // 2), rotation_rough, 1.0)
        cv2.warpAffine(middle, rot_mat, (middle.shape[1], middle.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, dst=rotated_middle, borderValue=-1.0)
        rotated_middle_crop = rotated_middle[50:-50, 50:-50]
        if False:
            view = ImageViewer(norm)
            ImageViewer(middle)
            ImageViewer(rotated_middle_crop)
            view.show()
            sys.exit()
    else:
        rotation_rough = 0
        norm = im.copy()
        f = {}
        ip.histogram_percentiles(norm, f)
        norm -= f['hist_percentile_01']
        norm /= (f['hist_percentile_90'] - f['hist_percentile_01'])
        pixel_ops.ApplyThresholdLT_F32(norm, norm, 0, 0)
        s5 = int(round(h * 0.4))
        middle_y, middle_x = h // 2, w // 2
        y1, y2 = max(0, middle_y - s5), min(middle_y + s5, h)
        x1, x2 = max(0, middle_x - s5), min(middle_x + s5, w)
        middle = norm[y1:y2, x1:x2]
        middle_h, middle_w = middle.shape
        rotated_middle_crop = norm[y1:y2, x1:x2]

    if False:
        view = ImageViewer(rotated_middle_crop)
        view.show()
        sys.exit()

    s = 20
    row_num = s + np.argmax(rotated_middle_crop.mean(axis=1)[s:-s])
    col_num = s + np.argmax(rotated_middle_crop.mean(axis=0)[s:-s])
    col = rotated_middle_crop[row_num - 10:row_num + 10, :].mean(axis=0)
    row = rotated_middle_crop[:, col_num - 10:col_num + 10].mean(axis=1)

    if im.shape[0] < 600:
        sigma = 2
    else:
        sigma = 4
    row_DoG = np.abs(ndimage.gaussian_filter1d(row, sigma=1) - ndimage.gaussian_filter1d(row, sigma=sigma))
    row_high_freq = np.median(row_DoG)
    col_DoG = np.abs(ndimage.gaussian_filter1d(col, sigma=1) - ndimage.gaussian_filter1d(col, sigma=sigma))
    col_high_freq = np.median(col_DoG)

    if min(row_high_freq, col_high_freq) < 0.00001:
        print("\nERROR: Can't compare horizontal and vertical frequency.")
        raise CellMissingException

    var_ratio = max(row_high_freq, col_high_freq) / min(row_high_freq, col_high_freq)

    # decide if fingers are perpendicular to BBs, or if they are a grid pattern
    if var_ratio < 2.0:
        # not much difference in high-frequency content. assume fingers make grid pattern (Line PERC)
        features['_fingers_grid'] = True
    else:
        # significant difference in high-freq.
        features['_fingers_grid'] = False

    def rotate():
        if not features['_fingers_grid']:
            # fingers in one direction, and should be horizontal
            return row_high_freq < col_high_freq
        elif module_cell:
            return True
        else:
            # fingers in both direction, so can't rely on high-freqency content
            # busbars should lead to largest peak in profiles
            row_dips = ndimage.gaussian_filter1d(row, sigma=1) - ndimage.gaussian_filter1d(row, sigma=10)
            col_dips = ndimage.gaussian_filter1d(col, sigma=1) - ndimage.gaussian_filter1d(col, sigma=10)

            return row_dips.min() < col_dips.min()

    if False:
        # print col_num
        print "Alg: %s, Row high freq: %0.03f, Col high freq: %0.03f, Ratio: %0.02f: Rotate: %s" % \
              (features['_alg_mode'], row_high_freq, col_high_freq, var_ratio, str(Rotate()))
        ImageViewer(rotated_middle_crop)
        plt.figure()
        if False:
            plt.plot(row, color='r', label="rows")
            plt.plot(col, color='b', label="cols")
        else:
            # plt.plot(row - ndimage.gaussian_filter1d(row, sigma=1), color='r')
            # plt.plot(col - ndimage.gaussian_filter1d(col, sigma=1), color='b')
            plt.plot(row_DoG, color='r', label="rows")
            plt.plot(col_DoG, color='b', label="cols")
        plt.legend()
        plt.show()
        sys.exit()

    if rotate():
        features['cell_rotated'] = True
        rotation_rough += 90
    else:
        features['cell_rotated'] = False

    if False:
        upright = np.empty_like(middle)
        rot_mat = cv2.getRotationMatrix2D((middle_w // 2, middle_h // 2), rotation_rough, 1.0)
        cv2.warpAffine(middle, rot_mat, (middle.shape[1], middle.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, dst=upright, borderValue=-1.0)
        view = ImageViewer(norm)
        view = ImageViewer(upright)
        view.show()
        sys.exit()

    if not already_cropped:
        #############
        # fine tune #
        #############
        # make fingers horizontal
        # - find some lines
        # - take median
        upright = np.empty_like(norm)
        rot_mat = cv2.getRotationMatrix2D((norm.shape[1] // 2, norm.shape[0] // 2), rotation_rough, 1.0)
        cv2.warpAffine(norm, rot_mat, (norm.shape[1], norm.shape[0]), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, dst=upright, borderValue=0.0)

        col_profile = ndimage.gaussian_filter1d(upright.mean(axis=0), sigma=5)
        left = np.argmax(col_profile[:len(col_profile) // 3])
        s = (2 * len(col_profile)) // 3
        right = s + np.argmax(col_profile[s:])

        if False:
            ImageViewer(upright)
            plt.figure()
            plt.plot(col_profile)
            plt.vlines([left, right], 0, 1)
            plt.show()

        col_left = upright[:, max(0, left - 3):left + 4].mean(axis=1)
        col_right = upright[:, right - 3:min(upright.shape[1] - 1, right + 4)].mean(axis=1)
        e = int(0.075 * w)
        idx = np.arange(h)
        mins_left = np.where((col_left < np.roll(col_left, 1)) &
                             (col_left < np.roll(col_left, -1)) &
                             (idx > e) & (idx < h - e))[0][::5]
        mins_right = np.where((col_right < np.roll(col_right, 1)) &
                              (col_right < np.roll(col_right, -1)))[0]
        if False:
            plt.figure()
            plt.plot(col_left)
            plt.plot(col_right)
            plt.plot(mins_left, col_left[mins_left], 'o')
            plt.plot(mins_right, col_right[mins_right], 'o')
            view = ImageViewer(norm)
            ImageViewer(upright)
            view.show()

        if len(mins_left) == 0 or len(mins_right) == 0:
            print("\nERROR: Cannot find any fingers.")
            raise CellMissingException

        # for each finger location on the left, find corresponding finger location on the right
        y_offsets = []
        lines = []
        vals = np.empty((right - left) * 2, np.float32)
        for e, finger_left_pos in enumerate(mins_left):
            right_closest_i = np.abs(mins_right - finger_left_pos).argmin()
            # right_closest_i = min(right_closest_i, len(mins_right)-2)
            if right_closest_i == 0 or right_closest_i == len(mins_right) - 1:
                continue

            rs = [mins_right[right_closest_i - 1], mins_right[right_closest_i], mins_right[right_closest_i + 1]]
            vs = []
            for r in rs:
                num_vals = pixel_ops.LineVals(upright, vals, finger_left_pos, left, r, right)
                vs.append(vals[:num_vals].mean())
            finger_right_pos = rs[vs.index(min(vs))]
            y_offsets.append(finger_right_pos - finger_left_pos)

            if False:
                print e, finger_left_pos, rs
                lines.append((finger_left_pos, left, finger_right_pos, right))

        if False:
            plt.figure()
            plt.plot(col_left, 'r')
            plt.plot(col_right, 'b')
            plt.plot(mins_left, col_left[mins_left], 'o')
            plt.plot(mins_right, col_right[mins_right], 'o')
            view = ImageViewer(norm)
            plt.figure()
            plt.imshow(upright, cmap=cm.gray)
            print len(lines)
            for line in lines:
                plt.plot([line[1], line[3]], [line[0], line[2]])
            view.show()

        rotation_correction = math.degrees(np.arctan2(np.median(y_offsets), right - left))
        rotation = rotation_rough + rotation_correction
    else:
        rotation = rotation_rough

    # output may not be square
    h, w = im.shape
    if features['cell_rotated']:
        dsize = (h, w)
        rot_mat = cv2.getRotationMatrix2D((w // 2, w // 2), rotation, 1.0)
    else:
        dsize = (w, h)
        rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, 1.0)

    rotated = np.empty((dsize[1], dsize[0]), np.float32)
    cv2.warpAffine(im, rot_mat, dsize, flags=cv2.INTER_LINEAR,
                   borderMode=cv2.BORDER_REPLICATE, dst=rotated)

    if False:
        view = ImageViewer(im)
        ImageViewer(rotated)
        view.show()

    features['crop_rotation'] = rotation

    return rotated


def main():
    pass


if __name__ == '__main__':
    main()
