import os, sys
import numpy as np
import features_slugs as slugs
import timeit
import image_processing as ip
from pprint import pprint
from image_processing import ImageViewer
import parameters
import pixel_ops
import cv2
import math
from scipy import optimize, ndimage, stats
from scipy.spatial import ConvexHull
import skimage
from skimage.transform import probabilistic_hough_line as probabilistic_hough
from skimage import draw
import matplotlib.pylab as plt

struct = ndimage.generate_binary_structure(2, 1)
circle_rr, circle_cc = draw.circle_perimeter(0, 0, 6)

DEFECT_UNLABELED = -1
DEFECT_NONE = 0
DEFECT_CRACK = 1
DEFECT_CHIP = 2
DEFECT_PINHOLE = 4
DEFECT_LABELS = ['None', 'Crack', 'Chip', 'Pinhole']


#################
# create_overlay #
#################

def create_overlay(features, folder_out=None):
    global defect_gt

    im_u8 = features['im_cropped_u8']
    im_rgb = np.empty((im_u8.shape[0], im_u8.shape[1], 3), np.float32)
    im_rgb[:, :, :] = im_u8[:, :, np.newaxis]

    if 'ov_dark_middle_u8' in features:
        impure = features["ov_dark_middle_u8"] // 2
        im_rgb[:, :, 0] += impure
        im_rgb[:, :, 1] -= impure
        im_rgb[:, :, 2] += impure

    if 'mk_slip_u8' in features:
        im_rgb[:, :, 0][features['mk_slip_u8'] == 1] = 0
        im_rgb[:, :, 1][features['mk_slip_u8'] == 1] = 128
        im_rgb[:, :, 2][features['mk_slip_u8'] == 1] = 0

    # add defects
    if 'mk_cracks_u8' in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_cracks_u8'], 'r')
    if 'mk_chips_u8' in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_chips_u8'], 'b')
    if 'mk_pinholes_u8' in features:
        im_rgb = ip.overlay_mask(im_rgb, features['mk_pinholes_u8'], 'g')

    im_rgb[im_rgb < 0] = 0
    im_rgb[im_rgb > 255] = 255
    im_rgb = im_rgb.astype(np.uint8)

    if False:
        # check if ground truth
        if features['fn'] in defect_gt:
            true_positives = np.zeros_like(features['mk_defects_u8'])
            for defect in defect_gt[features['fn']]:
                ((x1, y1), (x2, y2)) = defect['bb']
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                true_positives[y1:y2, x1:x2] = 1

                # check for false negative
                if (features['mk_defects_u8'][y1:y2, x1:x2] == defect['type']).sum() == 0:
                    im_rgb[y1:y2, x1, :] = (255, 0, 0)
                    im_rgb[y1:y2, x2, :] = (255, 0, 0)
                    im_rgb[y1, x1:x2, :] = (255, 0, 0)
                    im_rgb[y2, x1:x2, :] = (255, 0, 0)

            # check for false positives
            ccs, num_ccs = ip.connected_components(features['mk_defects_u8'] > 0)
            h, w = im_u8.shape
            for cc in range(1, num_ccs + 1):
                ys, xs = np.where(ccs == cc)
                y, x = int(round(ys.mean())), int(round(xs.mean()))
                if true_positives[y, x] == 1: continue
                y1, y2 = ys.min(), ys.max()
                x1, x2 = xs.min(), xs.max()
                y1 -= 2
                y2 += 2
                x1 -= 2
                x2 += 2
                y1 = max(0, y1)
                y2 = min(h - 1, y2)
                x1 = max(0, x1)
                x2 = min(w - 1, x2)
                im_rgb[y1:y2, x1, :] = (255, 0, 0)
                im_rgb[y1:y2, x2, :] = (255, 0, 0)
                im_rgb[y1, x1:x2, :] = (255, 0, 0)
                im_rgb[y2, x1:x2, :] = (255, 0, 0)

    return im_rgb


###############
# wafer_center #
###############
# @profile
def wafer_center(im, features):
    h, w = im.shape
    h2 = h // 2
    w2 = w // 2

    # find gradients
    if h > 600:
        sigma = 4.0
    else:
        sigma = 2.0
    smooth = cv2.GaussianBlur(im, (0, 0), sigma)
    edgesH = cv2.Sobel(smooth, cv2.CV_32F, 1, 0) * -1
    edgesV = cv2.Sobel(smooth, cv2.CV_32F, 0, 1) * -1
    edges = cv2.magnitude(edgesH, edgesV)
    r1 = w2 * 0.1
    r2 = w2 * 0.97
    pixel_ops.DistCenterRange(edges, r1, r2, w2, h2, 0)

    EDGE_THRESH = 0.025
    gradient_mask = edges >= EDGE_THRESH

    if False:
        view = ImageViewer(im)
        ImageViewer(edges)
        ImageViewer(gradient_mask)
        view.show()
        sys.exit()

    if gradient_mask.sum() < 100:
        features['wafer_middle_x'] = w2
        features['wafer_middle_y'] = h2
        dist_precent = 0
    else:
        # get a second point on the gradient line for each pixel in the mask
        ys, xs = np.mgrid[:h, :w]
        y1 = ys[gradient_mask]
        x1 = xs[gradient_mask]
        y2 = y1 + edgesV[gradient_mask] * 100
        x2 = x1 + edgesH[gradient_mask] * 100

        # find middle:
        #  - the gradient of a ring pixel will point towards the center
        #  - find the middle point that minimizes the average distance between
        #    each gradient line and candidate center
        A = x1 * y2
        B = x2 * y1
        Dx = x2 - x1
        Dy = y2 - y1
        denom = np.sqrt(Dx ** 2 + Dy ** 2)

        def AvgDist(args):
            y0, x0 = args
            dist = np.abs(Dy * x0 - Dx * y0 - A + B) / denom
            return dist.mean()

        t1 = timeit.default_timer()
        y_mid, x_mid = optimize.fmin(AvgDist, (h2, w2), disp=False)
        d_center = math.sqrt((h2 - y_mid) ** 2 + (w2 - x_mid) ** 2)
        y_mid, x_mid = int(round(y_mid)), int(round(x_mid))

        CZ_CENTER_THRESH = 7.5
        dist_precent = (d_center / (im.shape[0] / 2.0)) * 100

        if False:
            t2 = timeit.default_timer()
            print y_mid, x_mid
            print "Optimization time: %0.02f" % (t2 - t1)
            print "Distance between wafer center and image center: %0.02f" % (d_center)
            print "Reject: %s" % (dist_precent > CZ_CENTER_THRESH)
            if im[y_mid, x_mid] < 0.5:
                im[y_mid, x_mid] = 1
            else:
                im[y_mid, x_mid] = -1
            view = ImageViewer(im)
            view.show()
            sys.exit()

        if dist_precent > CZ_CENTER_THRESH:
            features['wafer_middle_x'] = w2
            features['wafer_middle_y'] = h2
            dist_precent = 0
        else:
            features['wafer_middle_x'] = x_mid
            features['wafer_middle_y'] = y_mid

    features['wafer_middle_dist'] = dist_precent


#######
# rds #
#######
def rds(cropped, features):
    radius = features['wafer_radius']

    # "ring defect strength" from Haunschild paper
    inner_edge = min(cropped.shape[0], cropped.shape[1]) / 2.0
    inner_mean, outer_mean = pixel_ops.RDS(cropped, radius, inner_edge, inner_edge)
    rds = outer_mean / inner_mean
    features['rds'] = rds

    # create a mask
    inner_val = max(0, min(255, int(round(inner_mean * 255))))
    outer_val = max(0, min(255, int(round(outer_mean * 255))))
    im_rds = np.ones_like(cropped, np.float32) * inner_val

    pixel_ops.ApplyThresholdGT_F32(features['im_center_dist_im'], im_rds, float(inner_edge), outer_val)
    pixel_ops.ApplyThresholdGT_F32(features['im_center_dist_im'], im_rds, features['wafer_radius'], 0)
    im_rds /= max(inner_val, outer_val)
    pixel_ops.ApplyThresholdGT_F32(im_rds, im_rds, 0.999, 0)
    im_rds *= 255
    features['ov_rds_u8'] = np.round(im_rds).astype(np.uint8)

    if False:
        view = ImageViewer(cropped)
        ImageViewer(features['ov_rds_u8'])
        view.show()

    # other rds metrics
    inner_edge = int(round(0.85 * radius))
    inner_mean, outer_mean = pixel_ops.RDS(cropped, radius, inner_edge, inner_edge)
    features['rds-15-85'] = outer_mean / inner_mean

    inner_edge = int(round(0.3 * radius))
    outer_edge = int(round(0.85 * radius))
    inner_mean, outer_mean = pixel_ops.RDS(cropped, radius, inner_edge, outer_edge)
    features['rds-15-30'] = outer_mean / inner_mean

    inner_edge = int(round(0.3 * radius))
    outer_edge = int(round(0.75 * radius))
    inner_mean, outer_mean = pixel_ops.RDS(cropped, radius, inner_edge, outer_edge)
    features['rds-25-30'] = outer_mean / inner_mean


##################
# ProfileMetrics #
##################
def radial_profile(cropped, features):
    h, w = cropped.shape
    radius = features['wafer_radius']
    if 'wafer_middle_y' in features:
        wafer_middle_y, wafer_middle_x = features['wafer_middle_y'], features['wafer_middle_x']
    else:
        wafer_middle_y, wafer_middle_x = h // 2, w // 2

    # calculate the average intensity for concentric circles around the wafer middle
    profile = np.zeros(cropped.shape[0], np.float32)
    counts = np.zeros(cropped.shape[0], np.float32)
    pixel_ops.RadialAverage(cropped, profile, counts, radius + 2,
                            wafer_middle_y, wafer_middle_x)
    profile[counts > 0] /= counts[counts > 0]
    # profile /= profile.max()
    profile = ndimage.gaussian_filter1d(profile, sigma=5)
    features['_radial_profile'] = profile

    # find the distance at which the corners become "dark"
    dark_thresh = 0.5
    dark_areas = np.where(profile > dark_thresh)[0]
    if len(dark_areas) > 0:
        radius_dark = dark_areas[-1]

        # width is the percentage of the dark area of the radius
        features['dark_corner_width'] = (radius - radius_dark) / float(radius)
        features['dark_corner_pixels'] = (radius - radius_dark)
        dark_percentage = counts[radius_dark:].sum() / float(counts.sum())
        features['dark_corner_area'] = dark_percentage

        # dark corner strength is an inverted rds using radius_dark
        inner_mean, outer_mean = pixel_ops.RDS(cropped, radius, radius_dark, radius_dark)
        features['dark_corner_strength'] = inner_mean / outer_mean

        corner_mask = np.ones_like(cropped, np.uint8)
        pixel_ops.ApplyThresholdLT_F32_U8(features['im_center_dist_im'], corner_mask, radius_dark, 0)
        pixel_ops.ApplyThresholdGT_F32_U8(features['im_center_dist_im'], corner_mask, radius, 0)
        features['mk_dark_corner_u8'] = corner_mask
    else:
        features['dark_corner_width'] = 0
        features['dark_corner_pixels'] = 0
        # features['dark_corners'] = 0
        features['dark_corner_area'] = 0
        features['dark_corner_strength'] = 0
        features['mk_dark_corner_u8'] = np.zeros_like(cropped, np.uint8)

    if False:
        ys, xs = draw.circle_perimeter(h // 2, w // 2, radius_dark)
        mask = ((ys >= 0) & (ys < h) & (xs >= 0) & (xs < w))
        circle = np.zeros_like(cropped, np.uint8)
        circle[ys[mask], xs[mask]] = 1
        circle = ndimage.binary_dilation(circle)
        ImageViewer(ip.overlay_mask(cropped, circle, 'r'))
        ImageViewer(corner_mask)
        plt.figure()
        plt.plot(profile)
        plt.show()

    # compute the mean intensity of the middle 10%
    features['middle_brightness'] = profile[:int(radius * 0.1)].mean()

    # rings cause bumps in profile. use variation as a
    #  limitation: if the rings are spirals, they can even themselves out at a given radius
    high_pass = (profile[:int(round(radius))] - ndimage.gaussian_filter1d(profile[:int(round(radius))], sigma=30))[
                10:-10]
    features['dark_rings_std'] = high_pass.std() * 100

    # get the slope from the center to the edge (not corner)
    edge_dist = (cropped.shape[0] // 2) - 10
    radial_average = profile[:edge_dist]
    xs = np.arange(len(radial_average))
    z = np.polyfit(xs, radial_average, 1)
    features['center_edge_slope'] = z[0] * 1000

    min_d = min(edge_dist, wafer_middle_y, wafer_middle_x, h - wafer_middle_y, w - wafer_middle_x)
    profiles = np.empty((4, min_d), np.float32)
    profiles[0, :] = ndimage.gaussian_filter1d(cropped[wafer_middle_y, wafer_middle_x:0:-1], sigma=5)[:min_d]
    profiles[1, :] = ndimage.gaussian_filter1d(cropped[wafer_middle_y, wafer_middle_x:], sigma=5)[:min_d]
    profiles[2, :] = ndimage.gaussian_filter1d(cropped[wafer_middle_y:0:-1, wafer_middle_x], sigma=5)[:min_d]
    profiles[3, :] = ndimage.gaussian_filter1d(cropped[wafer_middle_y:, wafer_middle_x], sigma=5)[:min_d]

    # general purpose asymmetry/outlier metrics

    # if l/r/t/b are all quite different there is an asymmetry, and that should be interesting
    features['radial_asymmetry_h'] = np.abs(profiles[0, :] - profiles[1, :]).mean() * 100
    features['radial_asymmetry_v'] = np.abs(profiles[2, :] - profiles[3, :]).mean() * 100

    # similar, but based on
    profile_col = cropped.mean(axis=0)
    profile_row = cropped.mean(axis=1)
    features['general_asymmetry'] = np.abs(profile_col - profile_col[::-1]).mean() + \
                                    np.abs(profile_row - profile_row[::-1]).mean()

    if False:
        view = ImageViewer(cropped)
        plt.figure()
        plt.plot(profile_col, 'r-')
        plt.plot(profile_row, 'g-')
        plt.plot(np.abs(profile_col - profile_col[::-1]), 'r-')
        plt.plot(np.abs(profile_row - profile_row[::-1]), 'g-')
        view.show()

    if False:
        print features['radial_asymmetry_h']
        print features['radial_asymmetry_v']
        ImageViewer(cropped)
        plt.figure()
        plt.plot(profiles[0, :], 'r-')
        plt.plot(profiles[1, :], 'r-')
        plt.plot(profiles[2, :], 'b-')
        plt.plot(profiles[3, :], 'b-')
        plt.plot(radial_average, 'g-')
        # plt.plot(diffs.mean(axis=0), 'g-')
        plt.show()

    if False:
        print features['middle_brightness']
        print features['dark_rings_std']
        print features['center_edge_slope']
        plt.figure()
        plt.plot(profile[:radius])
        plt.plot(ndimage.gaussian_filter1d(profile[:radius], sigma=30))
        plt.plot(high_pass)
        ImageViewer(cropped)
        plt.figure()
        plt.plot(profile[:radius + 2])
        plt.figure()
        plt.plot(xs, np.poly1d(z)(xs))
        plt.show()
        sys.exit()


###########################
# stripe_correction_rounded #
###########################
def stripe_correction_rounded(im, features):
    # im = im[:, ::-1]
    """
    This stripe correction is optimized for saw marks that are not straight
    """
    h, w = im.shape

    # find out saw marks are vertical or horizonal
    col_middle = im[:, (w // 2) - 5:(w // 2) + 6].mean(axis=1)[20:-20]
    row_middle = im[(h // 2) - 5:(h // 2) + 6, :].mean(axis=0)[20:-20]
    col_middle -= ndimage.gaussian_filter1d(col_middle, sigma=1)
    row_middle -= ndimage.gaussian_filter1d(row_middle, sigma=1)
    col_var = col_middle[20:-20].std()
    row_var = row_middle[20:-20].std()

    if False:
        print col_var, row_var
        plt.figure()
        plt.plot(col_middle)
        plt.plot(row_middle)
        plt.show()
        sys.exit()

    corrected = im.copy()
    s = 25
    f = np.ones((1, s), dtype=np.float32) / s
    filtered1 = ndimage.convolve(im, f, mode="reflect")
    corrected -= filtered1
    f = np.ones((s, 1), dtype=np.float32) / s
    filtered2 = ndimage.convolve(corrected, f, mode="reflect")
    corrected -= filtered2

    if False:
        view = ImageViewer(im)
        ImageViewer(filtered1)
        ImageViewer(filtered2)
        ImageViewer(corrected)
        view.show()

    # fix corners
    # - find spot where radius intersects the edge
    a = h / 2.0
    radius = features['wafer_radius']
    b = int(round(math.sqrt(radius ** 2 - a ** 2)))
    corner_len = int(round(a - b + 7))
    corner_len2 = corner_len + (s // 2) + 1

    s = 25
    if col_var > row_var:
        f_shape = (1, s)
    else:
        f_shape = (s, 1)

    # bottom left
    corner = im[-corner_len2:, :corner_len2]
    corner_med = ndimage.median_filter(corner, size=f_shape, mode="nearest")
    corrected[-corner_len:, :corner_len] = im[-corner_len:, :corner_len] - corner_med[-corner_len:, :corner_len]

    # top left
    corner = im[:corner_len2, :corner_len2]
    corner_med = ndimage.median_filter(corner, size=f_shape, mode="nearest")
    corrected[:corner_len, :corner_len] = im[:corner_len, :corner_len] - corner_med[:corner_len, :corner_len]

    # top right
    corner = im[:corner_len2, -corner_len2:]
    corner_med = ndimage.median_filter(corner, size=f_shape, mode="nearest")
    corrected[:corner_len, -corner_len:] = im[:corner_len, -corner_len:] - corner_med[:corner_len, -corner_len:]

    # bottom right
    corner = im[-corner_len2:, -corner_len2:]
    corner_med = ndimage.median_filter(corner, size=f_shape, mode="nearest")
    corrected[-corner_len:, -corner_len:] = im[-corner_len:, -corner_len:] - corner_med[-corner_len:, -corner_len:]

    if False:
        view = ImageViewer(im)
        ImageViewer(corrected)
        ImageViewer(corner)
        ImageViewer(corner_med)
        view.show()
        sys.exit()

    return corrected


##########################
# stripe_correction_skewed #
##########################
def stripe_correction_skewed(cropped, features):
    """
    This correction is designed for straight lines that aren't necessarily
     veritcal
    """
    # find rotation
    middleH = cropped[100:-100:2, 20:-20:2]
    middleV = cropped[20:-20:2, 100:-100:2]
    cols = middleH.sum(axis=0)
    cols -= ndimage.gaussian_filter1d(cols, sigma=20, mode="nearest")
    rows = middleV.sum(axis=1)
    rows -= ndimage.gaussian_filter1d(rows, sigma=20, mode="nearest")
    flip = False
    if rows.std() > cols.std():
        flip = True
        cropped = cropped.T

    if False:
        print "Flip: ", flip, rows.std(), cols.std()
        view = ImageViewer(middleH)
        view = ImageViewer(middleV)
        view = ImageViewer(cropped)
        view.show()
        sys.exit()

    # find skew
    middle = cropped[100:-100:2, 10:-10:2].copy()
    row_means = middle.mean(axis=1)
    middle -= row_means.reshape((middle.shape[0], 1))
    skews = np.linspace(-0.1, 0.1, 21, endpoint=True)
    avg_vars = []
    for s in skews:
        if abs(s) < 0.001:
            warped = middle
        else:
            if s >= 0.0:
                offset = int(s * middle.shape[0])
                M = np.array([[1, s, 0],
                              [0, 1, 0]], np.float32)
            else:
                offset = -int(s * middle.shape[0])
                M = np.array([[1, s, offset],
                              [0, 1, 0]], np.float32)
            warped = cv2.warpAffine(middle, M, (middle.shape[1] + offset, middle.shape[0]),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)[:, offset:-offset]

        means = warped.mean(axis=0)
        profile = means - ndimage.gaussian_filter1d(means, 4, mode="nearest")
        if False:
            plt.figure()
            plt.plot(means)
            plt.figure()
            plt.plot(profile)
            plt.show()

        if np.isnan(profile.var()):
            print 'WARNING #5', s
            avg_vars.append(0)
        else:
            avg_vars.append(profile.var())

    if False:
        print avg_vars
        plt.figure()
        plt.plot(avg_vars)
        plt.show()

    # skew cropped image
    min_i = np.argmax(avg_vars)
    if min_i == 0 or min_i == len(avg_vars) - 1 or max(avg_vars) < 1e-05:
        # ignore endpoints
        skew = 0
    else:
        skew = skews[min_i]

    if abs(skew) < 0.001:
        skewed = cropped
    else:
        M = np.array([[1, skew, 0],
                      [0, 1, 0]], np.float32)
        skewed = cv2.warpAffine(cropped, M, (cropped.shape[1], cropped.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        edge_zone = int(abs(skew) * cropped.shape[0] + 2)
        if skew < 0:
            skewed[:, -edge_zone:] = cropped[:, -edge_zone:]
        else:
            skewed[:, :edge_zone] = cropped[:, :edge_zone]

    if False:
        print skew
        view = ImageViewer(cropped)
        view = ImageViewer(skewed)
        view.show()
        sys.exit()

    # column correction
    tb = 150
    cols = skewed[tb:-tb, :].mean(axis=0)
    correction = cols.mean() - cols
    corrected = skewed + correction.reshape((1, correction.shape[0])).astype(np.float32)

    if False:
        print skew
        view = ImageViewer(cropped)
        view = ImageViewer(corrected)
        view.show()
        sys.exit()

    # undo skew
    if abs(skew) < 0.001:
        deskewed = corrected
    else:
        M = np.array([[1, -skew, 0],
                      [0, 1, 0]], np.float32)
        deskewed = cv2.warpAffine(corrected, M, (cropped.shape[1], cropped.shape[0]),
                                  flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    if False:
        view = ImageViewer(cropped)
        view = ImageViewer(skewed)
        view = ImageViewer(deskewed)
        view.show()
        sys.exit()

    if flip: deskewed = deskewed.T

    if not deskewed.flags['C_CONTIGUOUS']:
        deskewed = np.ascontiguousarray(deskewed)

    if False:
        # now correct other direction
        s = 100
        rows = deskewed[:, s:-s].mean(axis=1)
        correction = np.c_[rows.mean() - rows].astype(np.float32)
        final = deskewed + correction

    return deskewed


##########################
# stripe_correction_simple #
##########################
def stripe_correction_simple(cropped):
    h, w = cropped.shape
    s = h // 4
    cols = cropped[s:-s, :].mean(axis=0)
    corrected = cropped - cols.reshape((1, w)).astype(np.float32)

    rows = corrected[:, s:-s].mean(axis=1)
    corrected -= rows.reshape((h, 1)).astype(np.float32)

    return corrected


##############
# striations #
##############
def striations(cropped, features):
    polar = ip.polar_transform(cropped)[:, 50:]
    fl = 10
    f = np.ones((fl, 1), np.float32) / fl
    polar = ndimage.convolve(polar, f)
    shift = 5
    diff = polar - np.roll(polar, shift)
    diff[diff < 0.01] = 0
    diff[:, :shift] = 0
    features['striations'] = diff.sum()

    if False:
        print features['striations']
        view = ImageViewer(cropped)
        view = ImageViewer(polar)
        view = ImageViewer(diff)
        view.show()
        sys.exit()


#####################
# saw_mark_horizontal #
#####################
def saw_mark_horizontal(wafer):
    h = wafer.shape[0]
    h10 = h // 10
    h2 = h // 2

    sigma = 1
    strip_v = wafer[h10:-h10, h2 - 2:h2 + 3].mean(axis=1)
    strip_h = wafer[h2 - 2:h2 + 3, h10:-h10].mean(axis=0)
    strip_v -= ndimage.gaussian_filter1d(strip_v, sigma=sigma)
    strip_h -= ndimage.gaussian_filter1d(strip_h, sigma=sigma)

    if False:
        print strip_h.std(), strip_v.std()
        plt.figure()
        plt.plot(strip_v)
        plt.plot(strip_h)
        plt.show()

    return strip_v.std() > strip_h.std()


def fill_corners_edges(im, features, edge, corner_fill=None):
    h, w = im.shape
    radius = features['wafer_radius']

    if 'im_center_dist_im' in features:
        dist = features['im_center_dist_im']
    else:
        dist = features['im_center_dist']
    h2 = h // 2
    w2 = w // 2

    a = h / 2.0

    if radius < a:
        return im, 0
    b = int(round(math.sqrt(radius ** 2 - a ** 2)))
    corner_len = a - b + 5
    corner_avg = 0
    corner_filled = im.copy()

    # pixels to sample intensities along corners
    if corner_fill is None:
        ys, xs = draw.circle_perimeter(h // 2, w // 2, int(round(radius)) - edge)
        mask = ((ys >= 0) & (ys < h) & (xs >= 0) & (xs < w))
        ys = ys[mask]
        xs = xs[mask]
        corner_filled = im.copy()
        corner_avg = 0

    # top left
    if corner_fill is None:
        mask = ((ys < h2) & (xs < w2))
        corner_val = im[ys[mask], xs[mask]].mean()
    else:
        corner_val = corner_fill
    corner_avg += corner_val
    corner_filled[:corner_len, :corner_len][dist[:corner_len, :corner_len] > radius - edge] = corner_val

    # top right
    if corner_fill is None:
        mask = ((ys < h2) & (xs > w2))
        corner_val = im[ys[mask], xs[mask]].mean()
    else:
        corner_val = corner_fill
    corner_avg += corner_val
    corner_filled[:corner_len, -corner_len:][dist[:corner_len, -corner_len:] > radius - edge] = corner_val

    # bottom left

    if corner_fill is None:
        mask = ((ys > h2) & (xs < w2))
        corner_val = im[ys[mask], xs[mask]].mean()
    else:
        corner_val = corner_fill
    corner_avg += corner_val
    corner_filled[-corner_len:, :corner_len][dist[-corner_len:, :corner_len] > radius - edge] = corner_val

    # bottom right
    if corner_fill is None:
        mask = ((ys > h2) & (xs > w2))
        corner_val = im[ys[mask], xs[mask]].mean()
    else:
        corner_val = corner_fill
    corner_avg += corner_val
    corner_filled[-corner_len:, -corner_len:][dist[-corner_len:, -corner_len:] > radius - edge] = corner_val

    corner_avg /= 4.0

    # edges
    corner_filled[:, :edge] = np.c_[corner_filled[:, edge]]
    corner_filled[:, -edge:] = np.c_[corner_filled[:, -edge]]
    corner_filled[:edge, :] = np.r_[corner_filled[edge, :]]
    corner_filled[-edge:, :] = np.r_[corner_filled[-edge, :]]

    if False:
        r_mask = np.zeros_like(im, np.uint8)
        r_mask[ys, xs] = 1
        rgb = ip.overlay_mask(im, r_mask)
        view = ImageViewer(rgb)
        ImageViewer(corner_filled)
        view.show()
        sys.exit()

    return corner_filled, corner_avg


################
# process_rings #
################
# @profile
def process_rings(im, features, return_corrected=True):
    # call the slug alg to get rings strength
    im_rings = slugs.ring_strength(im, features)

    if False:
        view = ImageViewer(im)
        ImageViewer(im_rings)
        view.show()
        sys.exit()

    # smooth radiating outwards
    h2, w2 = features['wafer_middle_y'], features['wafer_middle_x']
    polar = ip.polar_transform(im_rings, center=(h2, w2))
    polar_sorted = np.sort(polar, axis=0)
    features['_radial_p30'] = polar_sorted[int(round(0.3 * polar.shape[0])), :]
    features['_radial_p50'] = polar_sorted[int(round(0.5 * polar.shape[0])), :]
    features['_radial_p70'] = polar_sorted[int(round(0.7 * polar.shape[0])), :]

    if not return_corrected:
        return

    # invert
    polar_smooth = ndimage.gaussian_filter1d(polar, sigma=15, axis=1, mode="reflect")
    radial_smooth = ip.polar_transform_inverse(im_rings, polar_smooth, center=(h2, w2))

    if False:
        view = ImageViewer(im_rings)
        ImageViewer(polar)
        ImageViewer(polar_sorted)
        ImageViewer(radial_smooth)
        plt.figure()
        plt.plot(features['_radial_p30'])
        plt.plot(features['_radial_p60'])
        view.show()
        sys.exit()

    correction = im_rings - radial_smooth
    edge = 5
    correction[:edge, :] = 0
    correction[-edge:, :] = 0
    correction[:, :edge] = 0
    correction[:, -edge:] = 0
    pixel_ops.ApplyThresholdGT_F32(features['im_center_dist_rot'], correction, features['wafer_radius'] - 20, 0)
    no_rings = im - correction

    # rings = np.abs(im-no_rings)
    # features['circle_strength_2'] = rings.mean()*1000-4

    if False:
        print features['circle_strength']
        print features['circle_strength_2']
        view = ImageViewer(im)
        ImageViewer(no_rings)
        view.show()
        sys.exit()

    return no_rings


################
# flatten_wafer #
################
def flatten_wafer(im, features):
    if parameters.STRIPE_CORRECTION == 0:
        corrected = im
    elif parameters.STRIPE_CORRECTION == 1:
        corrected = stripe_correction_simple(im)
    elif parameters.STRIPE_CORRECTION == 2:
        # assumes vertical, so find saw mark orientation
        flip = saw_mark_horizontal(im)
        if flip:
            im = im.T
        corrected = stripe_correction_skewed(im, features)
        if flip:
            corrected = np.ascontiguousarray(corrected.T)
    elif parameters.STRIPE_CORRECTION == 3:
        corrected = stripe_correction_rounded(im, features)
    else:
        print 'Unknown stripe correction: %s' % (parameters.STRIPE_CORRECTION)
        assert False

    if False:
        view = ImageViewer(im)
        view = ImageViewer(corrected)
        view.show()
        sys.exit()

    return corrected


crack_m_1 = [(0, 200), (0.035, 0.035)]
crack_m_2 = [(5, 9.8), (0.1, 0.035)]
crack_m_3 = [(9, 7.5), (0.12, 0.06)]


def classify_defect(defect_features):
    crack_size = defect_features['num_pixels']
    crack_strength = defect_features['strength_median']

    def discriminate(line):
        return (((crack_size - line[0][1]) * (line[1][0] - line[1][1])) -
                ((crack_strength - line[1][1]) * (line[0][0] - line[0][1]))) > 0

    classification = DEFECT_UNLABELED
    if (defect_features['strength_flat_max'] > 0.15 and defect_features['edge_dist'] < 4):
        classification = DEFECT_CHIP
    elif defect_features['num_pixels'] < 50 and defect_features['strength_depth'] > 0.3:
        classification = DEFECT_PINHOLE
    elif all([discriminate(crack_m_1), discriminate(crack_m_2), discriminate(crack_m_3)]):
        classification = DEFECT_CRACK
    else:
        classification = DEFECT_NONE

    return classification


###################
# crack_properties #
###################
def crack_properties(ys, xs, crack_num, features, defect_mask):
    # compute convex hull & minimum bounding rectangle
    ys_min, xs_min, ys_max, xs_max = ys.min(), xs.min(), ys.max(), xs.max()
    if ys_min == ys_max:
        ys_max += 1
        rect_len = xs_max - xs_min
        rect_width = 1.0
        rect_theta = 0.0
    elif xs_min == xs_max:
        xs_max += 1
        rect_len = ys_max - ys_min
        rect_width = 1.0
        rect_theta = 90.0
    else:
        points = np.array([xs, ys]).T
        cv_hull = ConvexHull(points)
        rect_len, rect_width, rect_theta = ip.min_rectanlge(points[cv_hull.vertices[::-1]])

    # save crack features
    features['defect%d_length' % (crack_num)] = rect_len
    features['defect%d_height' % (crack_num)] = rect_width
    features['defect%d_theta' % (crack_num)] = rect_theta
    features['defect%d_y' % (crack_num)] = ys.mean()
    features['defect%d_x' % (crack_num)] = xs.mean()
    features['defect%d_type' % (crack_num)] = defect_mask[ys[0], xs[0]]


##########
# cracks #
##########
def cracks(im, orig, features):
    global defect_gt
    h, w = im.shape

    # create a version with reflected borders
    im_pad = np.pad(im, ((2, 2), (2, 2)), mode="edge")

    if False:
        view = ImageViewer(im_pad)
        view.show()
        # sys.exit()

    # oriented filter to enhance cracks
    smoothed = cv2.GaussianBlur(im_pad, ksize=(0, 0), sigmaX=0.5, borderType=cv2.BORDER_REPLICATE)
    dark_lines = np.zeros(im_pad.shape, np.float32)
    pixel_ops.CrackEnhance(smoothed, dark_lines)
    smoothed = smoothed[2:-2, 2:-2]

    if False:
        view = ImageViewer(smoothed)
        view = ImageViewer(dark_lines)
        view.show()

    # only keep brightest responses
    num_vals = int(parameters.MAX_CRACK_AREA * im_pad.shape[0] * im_pad.shape[1])
    sorted_vals = np.sort(dark_lines.ravel())
    threshold = sorted_vals[-num_vals]
    candidates = np.zeros_like(im_pad, dtype=np.uint8)
    pixel_ops.ApplyThresholdGT_F32_U8(dark_lines, candidates, threshold, 1)

    # back to original size
    candidates = np.ascontiguousarray(candidates[2:-2, 2:-2])
    candidates[0, :] = 0
    candidates[-1, :] = 0
    candidates[:, 0] = 0
    candidates[:, -1] = 0

    if False:
        print threshold
        view = ImageViewer(orig)
        ImageViewer(im)
        ImageViewer(candidates)
        ImageViewer(dark_lines)
        view.show()
        sys.exit()

    # find biggest connected components
    ccs, num_ccs = ip.connected_components(candidates)
    cc_sizes = np.zeros(num_ccs + 1, np.int32)
    pixel_ops.CCSizes(ccs, cc_sizes)
    biggest_ccs = np.argsort(cc_sizes)[::-1]

    if False:
        print threshold
        print cc_sizes[573]
        view = ImageViewer(orig)
        ImageViewer(ccs)
        view.show()
        sys.exit()

    if False:
        # initialize crack features to 0
        # NOTE: this is now done in RunWaferCZ
        for c in range(1, parameters.CRACK_DETAILS + 1):
            for feature in ['defect%d_length', 'defect%d_height', 'defect%d_theta',
                            'defect%d_y', 'defect%d_x', 'defect%d_type']:
                features[feature % (c)] = 0

    defect_mask = np.zeros_like(im, np.uint8)
    num_top = parameters.MAX_NUM_CRACKS
    defect_found = False
    middle_y, middle_x = features['wafer_middle_y'], features['wafer_middle_y']
    for c in range(num_top):
        cc_size = cc_sizes[biggest_ccs[c]]
        if 'tuning' not in features and cc_size < parameters.MIN_CZ_CRACK_SIZE: continue
        defect_features = {}
        ys, xs = np.where(ccs == biggest_ccs[c])
        y = int(round(ys.mean()))
        x = int(round(xs.mean()))

        # find the crack and its outline
        window_y1, window_x1 = max(0, ys.min() - 2), max(0, xs.min() - 2)
        window_y2, window_x2 = min(h, ys.max() + 3), min(w, xs.max() + 3)
        mask_orig = orig[window_y1:window_y2, window_x1:window_x2]
        mask_flat = im[window_y1:window_y2, window_x1:window_x2]
        defect_pixels = (ccs[window_y1:window_y2, window_x1:window_x2] == biggest_ccs[c])
        mask_crack2 = ndimage.binary_dilation(defect_pixels, struct, iterations=1)
        mask_crack3 = ndimage.binary_dilation(mask_crack2, struct, iterations=1)
        defect_outline = mask_crack3.astype(np.uint8) - mask_crack2.astype(np.uint8)

        if False:
            view = ImageViewer(mask_flat)
            ImageViewer(defect_pixels)
            ImageViewer(defect_outline)
            view.show()

        # compute some features of the defect that will be used for classification
        defect_features['strength_median'] = np.median(mask_orig[defect_outline]) - np.median(mask_orig[defect_pixels])
        defect_features['strength_mean'] = mask_orig[defect_outline].mean() - mask_orig[defect_pixels].mean()
        defect_features['strength_median_flat'] = np.median(mask_flat[defect_outline]) - np.median(
            mask_flat[defect_pixels])
        defect_features['strength_mean_flat'] = mask_flat[defect_outline].mean() - mask_flat[defect_pixels].mean()
        defect_features['strength_flat_max'] = mask_flat.min() * -1
        defect_features['num_pixels'] = cc_size
        defect_features['edge_dist'] = min(x, y, w - 1 - x, h - 1 - y)
        defect_features['aspect_ratio'] = (max(ys.max() - ys.min(), xs.max() - xs.min()) /
                                           float(max(1, min(ys.max() - ys.min(), xs.max() - xs.min()))))
        defect_features['fill_ratio'] = defect_pixels.sum() / float(defect_pixels.shape[0] * defect_pixels.shape[1])
        defect_features['location_y'] = y
        defect_features['location_x'] = x

        if defect_features['edge_dist'] < 10:
            # ring removal doesn't work well along the edges, so filter out
            #  cracks that are arcs around the center
            # - standard deviation of pixel distance to center
            coords = np.where(defect_pixels)
            ys = np.array(coords[0])
            xs = np.array(coords[1])
            ys += window_y1
            xs += window_x1
            dists = np.sqrt((ys - middle_y) ** 2 + (xs - middle_x) ** 2)

            if False:
                print dists.std()
                view = ImageViewer(mask_flat)
                ImageViewer(defect_pixels)
                ImageViewer(im)
                ImageViewer(defect_outline)
                view.show()

            if dists.std() < 0.7: continue

        # pinhole features: find difference between local minimum and surrounding circle
        w2_y1, w2_x1 = max(0, ys.min() - 5), max(0, xs.min() - 5)
        w2_y2, w2_x2 = min(h, ys.max() + 6), min(w, xs.max() + 6)
        (y_min, x_min) = ndimage.minimum_position(smoothed[w2_y1:w2_y2, w2_x1:w2_x2])
        y_min += w2_y1
        x_min += w2_x1
        circle_ys = circle_rr + y_min
        circle_xs = circle_cc + x_min
        if all([circle_ys.min() >= 0, circle_ys.max() < h,
                circle_xs.min() >= 0, circle_xs.max() < w]):
            defect_features['strength_depth'] = orig[circle_ys, circle_xs].mean() - orig[y_min, x_min]
        else:
            defect_features['strength_depth'] = 0

        if False:
            # check ground truth
            fn_id = os.path.split(features['fn'])[1]
            defect_features['ground_truth'] = DEFECT_NONE
            for crack_dict in defect_gt[fn_id]:
                ((x1, y1), (x2, y2)) = crack_dict['bb']
                if ((min(x1, x2) < x < max(x1, x2)) and
                        (min(y1, y2) < y < max(y1, y2))):
                    defect_features['ground_truth'] = crack_dict['type']
                    break

            # show crack
            orig[int(ys.mean()), int(xs.mean())] = 2
            pprint(defect_features)
            view = ImageViewer(orig)
            ImageViewer(mask_orig, vmin=0, vmax=1)
            ImageViewer(defect_pixels)
            ImageViewer(defect_outline)
            view.show()

        if 'tuning' in features:
            fn = features['fn_features']
            features = sorted(defect_features.keys())
            feature_list = ','.join(features)
            if not os.path.isfile(fn):
                with open(fn, "w") as f:
                    f.write("filename,%s\n" % (feature_list))

            with open(fn, "a") as f:
                f.write("%s" % (features['fn']))
                for feature in features:
                    f.write(',%0.05f' % (defect_features[feature]))
                f.write("\n")

        # classify
        classification = classify_defect(defect_features)
        if classification in [DEFECT_NONE, DEFECT_UNLABELED]:
            continue
        defect_found = True
        defect_mask[window_y1:window_y2, window_x1:window_x2] = defect_pixels * classification

    if False:
        view = ImageViewer(im)
        ImageViewer(defect_mask)
        view.show()

    num_cracks = 0
    if defect_found:
        # the filter doesn't find areas with high curvature, intersections or pinhole
        #  middles, so apply an isotropic filter to fill in anything missing
        high_pass = np.zeros_like(im, np.uint8)
        high_pass_filter = (cv2.GaussianBlur(orig, ksize=(0, 0), sigmaX=3.0,
                                             borderType=cv2.BORDER_REPLICATE) - orig)
        high_pass[high_pass_filter > 0.035] = 1
        high_pass[:2, :] = 0
        high_pass[-2:, :] = 0
        high_pass[:, :2] = 0
        high_pass[:, -2:] = 0

        hp_ccs, _ = ip.connected_components(high_pass)

        if False:
            view = ImageViewer(orig)
            view = ImageViewer(defect_mask)
            ImageViewer(high_pass_filter)
            ImageViewer(high_pass)
            view.show()

        # get the CCs from the HP mask that touch a crack
        for defect_type in [DEFECT_CHIP, DEFECT_CRACK, DEFECT_PINHOLE]:
            ccs = np.unique(hp_ccs[defect_mask == defect_type])
            if ccs.shape[0] >= 0:
                ccs = ccs[ccs != 0]
            if ccs.shape[0] >= 0:
                defect_mask[np.in1d(hp_ccs.ravel(), ccs).reshape(hp_ccs.shape)] = defect_type

        if False:
            view = ImageViewer(orig)
            ImageViewer(defect_mask)
            view.show()
            sys.exit()

        # we now have defects - compute properties
        ccs, num_cracks = ip.connected_components(defect_mask > 0)
        cc_sizes = np.zeros(num_cracks + 1, np.int32)
        pixel_ops.CCSizes(ccs, cc_sizes)
        biggest_ccs = np.argsort(cc_sizes)[::-1]

        for c in range(min(num_cracks, parameters.MAX_NUM_CRACKS)):
            crack_label = biggest_ccs[c]
            ys, xs = np.where(ccs == crack_label)

            if ('input_param_verbose' not in features or features['input_param_verbose'] or crack_label <= 5):
                crack_properties(ys, xs, crack_label, features, defect_mask)

    features['defect_count'] = num_cracks
    if num_cracks > 0:
        features['defect_present'] = 1
    else:
        features['defect_present'] = 0
    features['defect_length'] = pixel_ops.CountEqual_U8(defect_mask, 1)
    # features['mask_defects_u8'] = defect_mask
    features['mk_cracks_u8'] = (defect_mask == DEFECT_CRACK).astype(np.uint8)
    features['mk_chips_u8'] = (defect_mask == DEFECT_CHIP).astype(np.uint8)
    features['mk_pinholes_u8'] = (defect_mask == DEFECT_PINHOLE).astype(np.uint8)

    if False:
        pprint(features)
        view = ImageViewer(im)
        ImageViewer(ccs)
        ImageViewer(features['mk_defects_u8'])
        view.show()
        # sys.exit()

    return


#############
# slip_lines #
#############
def slip_lines(im, features, filter_length, filter_width, percentile):
    # enhance diagonal lines
    filtered = np.zeros_like(im, np.float32)
    scratch = np.empty((6, filter_length), np.float32)
    pixel_ops.SlipEnhance(im, filtered, (filter_length - 1) // 2, filter_width, scratch)

    # threshold top X percent
    thresh = stats.scoreatpercentile(filtered[::2, ::2].flat, percentile)
    slip_lines = np.zeros_like(im, dtype=np.uint8)
    slip_lines[filtered >= thresh] = 1

    # join candidates into lines
    lines = probabilistic_hough(slip_lines, threshold=1, line_length=15, line_gap=3,
                                theta=np.array([math.radians(45), math.radians(135)]))
    slippers = np.zeros_like(filtered, np.uint8)
    for _, line in enumerate(lines):
        ys, xs = skimage.draw.line(line[0][1], line[0][0],
                                   line[1][1], line[1][0])
        slippers[ys, xs] = 1

    # ignore any line that doesn't have an end that is close to wafer edge
    cc_lables, num_lines = ip.connected_components(slippers)
    distances = np.zeros(num_lines + 1, np.float32)
    pixel_ops.SlipperDistances(slippers, cc_lables, distances)

    if num_lines >= 1:
        features['slip_ds'] = distances[1:].max()
    else:
        features['slip_ds'] = 0

    if (num_lines >= parameters.MIN_SLIP_NUM and
                features['slip_ds'] >= parameters.MIN_SLIP_DIST):
        features['slip_score'] = slippers.sum()
        features['slip_count'] = num_lines
        features['mk_slip_u8'] = cv2.dilate(slippers, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        features['slip_lines_present'] = True
    else:
        features['slip_score'] = 0
        features['slip_count'] = 0
        features['mk_slip_u8'] = np.zeros_like(slip_lines)
        features['slip_lines_present'] = False

    if False:
        print features['slip_score']
        print features['slip_count']
        view = ImageViewer(im)
        ImageViewer(filtered)
        ImageViewer(slip_lines)
        ImageViewer(features['mk_slip_u8'])
        view.show()
        sys.exit()


##############
# dark_middle #
##############
def dark_middle(im, features):
    radius = features['wafer_radius']
    if features['_alg_mode'] == 'mono cell':
        # for cells, bright areas complicate matters
        #  - higher variance means should use lower percentile
        p70 = features['_radial_p70'] / features['_radial_p70'].max()
        p30 = features['_radial_p30'] / features['_radial_p70'].max()
        p50 = features['_radial_p50'] / features['_radial_p70'].max()
        variance = p70 - p30

        if True:
            radial_profile = p50
            # high_var = np.where(variance > 0.6)[0]
            high_var = np.where(variance > 0.2)[0]
            if len(high_var) > 0:
                radial_profile[high_var[0]:] = radial_profile[high_var[0]]
        else:
            # blend based on variance. as variance increases, move towards lower
            blend = variance.copy()
            thresh = 0.5
            blend[blend > thresh] = thresh
            blend /= thresh
            radial_profile = blend * features['_radial_p70'] + (1 - blend) * features['_radial_p30']

        if False:
            ImageViewer(im)
            plt.figure()
            plt.plot(p30, 'r')
            plt.plot(p70, 'g')
            plt.plot(variance, 'b')
            plt.plot(radial_profile, 'k')
            # plt.plot(blend, 'y')
            plt.show()
    else:
        radial_profile = features['_radial_p30']
    radial_profile /= radial_profile.max()
    x = np.where(radial_profile == 1.0)[0][0]
    radial_profile[x:] = 1.0

    if False:
        ImageViewer(im)
        # ImageViewer(features['im_rings'])
        plt.figure()
        plt.plot(radial_profile)
        plt.show()

    # skip small areas (likely artefect due to busbar in the middle)
    dark_rs = np.where(radial_profile < 0.8)[0]
    if len(dark_rs) == 0:
        radial_profile = np.ones_like(radial_profile)
    else:
        dark_r_dist = np.where(radial_profile < 0.8)[0][-1] / float(radius)
        if dark_r_dist < 0.2:
            radial_profile = np.ones_like(radial_profile)

    # threshold to ignore dark middles
    dark_middle_strength = 1.0 - radial_profile[0]
    if dark_middle_strength < parameters.CELL_DARK_MIDDLE_MIN:
        radial_profile = np.ones_like(radial_profile)
        dark_middle_strength = 0

    scale = parameters.CELL_DARK_MIDDLE_SCALE
    radial_profile = (1.0 - scale) + (radial_profile * scale)

    r = np.round(features['im_center_dist_rot']).astype(np.int32)
    r[r > len(radial_profile) - 1] = len(radial_profile) - 1
    im_impure = 1 - np.take(radial_profile, r)
    features['ov_dark_middle_u8'] = (im_impure * 255).astype(np.uint8)
    features['dark_middle_strength'] = dark_middle_strength
    features['dark_middle_area_fraction'] = (im_impure > 0.3).mean()

    if False:
        print features['dark_middle_area_fraction']
        plt.figure()
        plt.plot(radial_profile)
        ImageViewer(im)
        ImageViewer(im_impure)
        ImageViewer(im_impure > 0.3)
        plt.show()

    return


#####################
# feature_extraction #
#####################
# @profile
def feature_extraction(cropped, crop_props, features):
    h, w = cropped.shape
    features['wafer_radius'] = crop_props['radius']
    features['crop_rotation'] = crop_props['estimated_rotation']
    if not cropped.flags['C_CONTIGUOUS']:
        cropped = np.ascontiguousarray(cropped)

    if False:
        print crop_props.keys()
        print features.keys()
        view = ImageViewer(cropped)
        view.show()
        sys.exit()

    ip.histogram_percentiles(cropped, features, h // 2, w // 2, features['wafer_radius'])
    cropped[cropped > features['hist_percentile_99']] = features['hist_percentile_99']

    # normalise image
    min_val = features['hist_percentile_01'] / float(features['hist_percentile_99'])
    norm_upper = features['hist_percentile_99']
    norm_lower = min(0.2, min_val)
    normed = ((cropped / norm_upper) - norm_lower) / (1 - norm_lower)

    if False:
        view = ImageViewer(cropped)
        ImageViewer(normed)
        view.show()
        sys.exit()

    # find the wafer center, which is the point around which the wafer
    #  was rotated during manufacturing. note that this may be different from
    #  the image center
    wafer_center(normed, features)

    # calculate distance from wafer rotation middle
    r, theta = np.empty_like(normed, np.float32), np.empty_like(normed, np.float32)
    pixel_ops.CenterDistance(r, theta, features['wafer_middle_y'], features['wafer_middle_x'])
    features['im_center_dist_rot'] = r
    features['im_center_theta_rot'] = theta

    # calculate distance from midpoint
    if h // 2 != features['wafer_middle_y'] or w // 2 != features['wafer_middle_x']:
        r, theta = np.empty_like(cropped, np.float32), np.empty_like(cropped, np.float32)
        pixel_ops.CenterDistance(r, theta, h // 2, w // 2)
        features['im_center_dist_im'] = r
        features['im_center_theta_im'] = theta
    else:
        features['im_center_dist_im'] = features['im_center_dist_rot']
        features['im_center_theta_im'] = features['im_center_theta_rot']

    if False:
        middle_mask = np.zeros_like(normed, np.uint8)
        middle_mask[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 1
        rgb = ip.overlay_mask(normed, middle_mask, colour='r')
        middle_mask[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3] = 0
        y, x = features['wafer_middle_y'], features['wafer_middle_x']
        middle_mask[y - 1:y + 2, x - 1:x + 2] = 1
        rgb = ip.overlay_mask(rgb, middle_mask, colour='b')
        view = ImageViewer(rgb)
        view.show()
        sys.exit()

    # create mask: 1=background
    wafer_mask = np.zeros_like(cropped, np.uint8)
    pixel_ops.ApplyThresholdGT_F32_U8(features['im_center_dist_im'], wafer_mask, features['wafer_radius'], 1)
    features['bl_cropped_u8'] = wafer_mask

    if False:
        view = ImageViewer(cropped)
        ImageViewer(wafer_mask)
        view.show()

    if parameters.MONO_WAFER_HISTOGRAM_ONLY:
        features['mk_defects_u8'] = np.zeros_like(normed, np.uint8)
        features['ov_dark_middle_u8'] = np.zeros_like(normed, np.uint8)
    elif 'input_param_skip_features' in features and int(features['input_param_skip_features']) == 1:
        pass
    else:
        # ring detection and correction
        no_rings = process_rings(normed, features)

        # create a version that is "flat", which has gradients, saw marks, etc removes
        flat = flatten_wafer(no_rings, features)

        # create a version that has rings removed
        radial_profile(normed, features)

        # compute some metrics without stripe correction
        rds(normed, features)
        dark_middle(normed, features)
        if parameters.DETECT_STRIATIONS:
            striations(normed, crop_props, features)

        cracks(flat, normed, features)
        if parameters.DETECT_SLIP_LINES:
            slip_lines(no_rings, features, 9, 3, 99.0)

    features['im_cropped_u8'] = (np.clip(normed, 0.0, 1.0) * 255).astype(np.uint8)
    if cropped.dtype.type is np.uint16:
        features['im_cropped_u16'] = cropped
    else:
        features['im_cropped_u16'] = cropped.astype(np.uint16)

    return


def main():
    pass


if __name__ == "__main__":
    main()
