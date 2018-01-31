import sys
import numpy as np
import parameters
import image_processing as ip
import cropping
from image_processing import ImageViewer
import features_cz_wafer as cz_wafer
import pixel_ops
import cv2
import math
from scipy import ndimage, interpolate
import matplotlib.pylab as plt
from skimage import draw
from timeit import default_timer


def create_overlay(im, features):
    h, w = im.shape
    if 'center_y' in features:
        crop_mask = np.zeros_like(im, np.uint8)
        y = int(round(features['center_y']))
        x = int(round(features['center_x']))
        r = int(round(features['radius']))
        rr, cc = draw.circle_perimeter(y, x, r)
        mask = ((rr < 0) | (rr >= h) | (cc < 0) | (cc >= w))
        rr = rr[~mask]
        cc = cc[~mask]
        crop_mask[rr, cc] = 1
        rgb = ip.overlay_mask(im, crop_mask)
    else:
        rgb = im

    return rgb


def find_slug(im, features):
    h, w = im.shape
    h2, w2 = h // 2, w // 2

    # highlight edges in each quadrant
    edgesH = cv2.Sobel(im, cv2.CV_32F, 0, 1)
    edgesV = cv2.Sobel(im, cv2.CV_32F, 1, 0)
    corner_edges = np.zeros_like(im)
    corner_edges[:h2, :w2] = edgesH[:h2, :w2] + edgesV[:h2, :w2]
    corner_edges[:h2, -w2:] = edgesH[:h2, -w2:] - edgesV[:h2, -w2:]
    corner_edges[-h2:, -w2:] = -1 * edgesH[-h2:, -w2:] - edgesV[-h2:, -w2:]
    corner_edges[-h2:, :w2] = -1 * edgesH[-h2:, :w2] + edgesV[-h2:, :w2]

    # find points on the corners
    left = corner_edges[:, :w2]
    ys = np.arange(left.shape[0])
    xs = np.argmax(left, axis=1)
    mask = corner_edges[ys, xs] > 0.4
    ys = ys[mask]
    xs = xs[mask]
    right = corner_edges[:, w2:]
    ys2 = np.arange(right.shape[0])
    xs2 = w2 + np.argmax(right, axis=1)
    mask = corner_edges[ys2, xs2] > 0.4
    ys2 = ys2[mask]
    xs2 = xs2[mask]
    ys = np.r_[ys, ys2]
    xs = np.r_[xs, xs2]

    if False:
        ImageViewer(corner_edges)
        plt.figure()
        plt.imshow(im, cmap="gray")
        plt.plot(xs, ys, "o")
        plt.show()
        sys.exit()

    t1 = default_timer()

    # user Hough transform to vote on most likely center/radius
    # - assume true center is within 150 pixels of image middle

    # phrase 1: rough fit
    MAX_OFFSET = 200
    step = 3
    acc_ys = np.arange(h2 - MAX_OFFSET, h2 + MAX_OFFSET + 1, step)
    acc_xs = np.arange(w2 - MAX_OFFSET, w2 + MAX_OFFSET + 1, step)
    diag = math.sqrt(h2 ** 2 + w2 ** 2)
    min_r = int(0.5 * diag)
    max_r = int(diag)
    acc = np.zeros((acc_ys.shape[0], acc_xs.shape[0], max_r - min_r), np.int32)
    pixel_ops.CircleHoughAcc2(ys, xs, acc_ys, acc_xs, acc, min_r, max_r)
    acc = ndimage.gaussian_filter(acc.astype(np.float32), sigma=(1, 1, 0))
    i, j, r = ndimage.maximum_position(acc)
    middle_y, middle_x, radius = acc_ys[i], acc_xs[j], r + min_r

    if True:
        # phrase 2: fine tune
        acc_ys = np.arange(middle_y - (2 * step), middle_y + (2 * step) + 1)
        acc_xs = np.arange(middle_x - (2 * step), middle_x + (2 * step) + 1)
        min_r = int(radius - 10)
        max_r = int(radius + 10)
        acc = np.zeros((acc_ys.shape[0], acc_xs.shape[0], max_r - min_r), np.int32)
        pixel_ops.CircleHoughAcc2(ys, xs, acc_ys, acc_xs, acc, min_r, max_r)
        acc = ndimage.gaussian_filter(acc.astype(np.float32), sigma=(1, 1, 0))
        i, j, r = ndimage.maximum_position(acc)

        middle_y, middle_x, radius = acc_ys[i], acc_xs[j], r + min_r

    features['center_y'] = middle_y
    features['center_x'] = middle_x
    features['radius'] = radius
    features['crop_rotation'] = 0
    features['crop_left'] = 0
    features['crop_right'] = im.shape[1] - 1
    features['crop_top'] = 0
    features['crop_bottom'] = im.shape[0] - 1

    mask = np.zeros_like(im, np.uint8)
    r, theta = np.empty_like(im, np.float32), np.empty_like(im, np.float32)
    pixel_ops.CenterDistance(r, theta, middle_y, middle_x)
    pixel_ops.ApplyThresholdGT_F32_U8(r, mask, radius, 1)

    features['bl_uncropped_u8'] = mask
    features['bl_cropped_u8'] = mask

    if False:
        print default_timer() - t1
        rgb = create_overlay(im, features)
        view = ImageViewer(rgb)
        # ImageViewer(mask)
        view.show()
        sys.exit()


def rds(im, features):
    # "ring defect strength" from Haunschild paper
    # features['param_rds_percent'] = 0.9
    if 'param_rds_percent' in features:
        if features['param_rds_percent'] > 1:
            features['param_rds_percent'] = features['param_rds_percent'] / 100.0

        # user-specified value (for now this is a percentage of radius)
        inner_edge = features['param_rds_percent'] * features['radius']
    else:
        # by default, half and half
        inner_edge = features['radius'] / 2.0

    inner_mean, outer_mean = pixel_ops.RDSlug(im, features['radius'], inner_edge,
                                              features['center_y'], features['center_x'])
    rds = outer_mean / inner_mean
    features['rds'] = rds


def radial_profile(im, features):
    h, w = im.shape
    if False:
        # plt.figure()
        # for r in range(0, 360, 30):
        #     x_diff = int(math.sin(math.radians(r))*1000 + features['center_x'])
        #     y_diff = int(math.cos(math.radians(r))*1000 + features['center_y'])
        #     ys, xs = draw.line(features['center_y'], features['center_x'],
        #                        x_diff, y_diff)
        #     mask = ((xs >=0) & (xs < w) & (ys >= 0) & (ys < h))
        #     ys = ys[mask]
        #     xs = xs[mask]
        #
        #     #im[ys, xs] = 0
        #     plt.plot(ndimage.gaussian_filter1d(im[ys, xs], sigma=3))
        view = ImageViewer(im)
        view.show()
        sys.exit()

    # calculate the average intensity for concentric circles around the wafer middle
    profile = np.zeros(im.shape[0], np.float32)
    counts = np.zeros(im.shape[0], np.float32)
    pixel_ops.RadialAverage(im, profile, counts, features['radius'],
                            features['center_y'],
                            features['center_x'])
    profile[counts > 0] /= counts[counts > 0]
    profile /= profile.max()

    # find the distance at which the corners become "dark"
    dark_thresh = 0.5
    radius_dark = np.where(profile > dark_thresh)[0][-1]

    # width is the percentage of the dark area of the radius
    features['dark_corner_width'] = (features['radius'] - radius_dark) / float(features['radius'])
    if features['dark_corner_width'] < 0.0075:
        features['dark_corner_width'] = 0
        features['dark_corner_pixels'] = 0
        features['dark_corner_strength'] = 0
        features['dark_corner_area'] = 0
    else:
        features['dark_corner_pixels'] = (features['radius'] - radius_dark)

        # area is the percentage of the slug surface that is a dark corner
        area_whole = math.pi * features['radius'] * features['radius']
        area_inner = math.pi * radius_dark * radius_dark
        features['dark_corner_area'] = (area_whole - area_inner) / float(area_whole)

        # dark corner strength
        # - just do an inverted rds with radius_dark
        inner_mean, outer_mean = pixel_ops.RDSlug(im, features['radius'], radius_dark,
                                                  features['center_y'], features['center_x'])
        features['dark_corner_strength'] = inner_mean / outer_mean

    # compute the mean intensity of the middle 10%
    features['middle_brightness'] = profile[:int(features['radius'] * 0.1)].mean()

    # bright corners
    # find segments above 0.8
    bright_regions = profile > 0.8
    labeled_array, num_features = ndimage.label(bright_regions)
    features['bright_corner_width'] = 0
    features['bright_corner_pixels'] = 0
    features['bright_corner_strength'] = 0
    # pick outer most
    if num_features > 0:
        outer_label = num_features
        # make sure end is close to edge
        start, stop = np.where(labeled_array == outer_label)[0][[0, -1]]
        end_percent = stop / float(features['radius'])
        if end_percent > 0.95:
            features['bright_corner_width'] = (stop - start) / float(features['radius'])
            features['bright_corner_pixels'] = (stop - start)
            if start > 0 and stop - start > 0:
                features['bright_corner_strength'] = profile[start:stop].mean() / float(profile[:start].mean())
            else:
                features['bright_corner_strength'] = 0
    if features['bright_corner_width'] > 0.9:
        features['bright_corner_width'] = 0
        features['bright_corner_pixels'] = 0

    if False:
        # import matplotlib.pylab as plt
        import matplotlib.cm as cm
        print features['dark_corner_area']
        print features['middle_brightness']
        print features['bright_corner_width']
        print features['bright_corner_strength']
        ImageViewer(im)
        plt.figure()
        plt.plot(profile[:features['radius'] + 2])
        plt.show()
        # sys.exit()


def feature_extraction(im, features, skip_features=False):
    # median filter to remove noise
    im = cv2.medianBlur(im, 3)
    h, w = im.shape

    # normalize
    hist_features = {}
    ip.histogram_percentiles(im, hist_features)
    norm = im / hist_features['hist_percentile_99.9']
    pixel_ops.ClipImage(norm, 0, 1)

    # automatically determine if square or round
    is_round = True
    crop_props = None
    try:
        # try cropping using wafer alg
        # im = np.ascontiguousarray(im[::-1, :])
        crop_props = cropping.crop_wafer_cz(im, create_mask=True, output_error=False)

        # if round, rotation will likely be high
        if abs(crop_props['estimated_rotation']) < 5:
            # make sure most of foreground mask is actually foreground
            f = {}
            ip.histogram_percentiles(im, f)
            norm = im / f['hist_percentile_99.9']
            coverage = (norm[crop_props['mask'] == 0] > 0.5).mean()
            if coverage > 0.97:
                is_round = False
            if False:
                print coverage
                view = ImageViewer(norm)
                ImageViewer(crop_props['mask'])
                view.show()
    except:
        pass

    if False:
        print "Is round:", is_round
        view = ImageViewer(im)
        view.show()

    if is_round:
        # find center and radius
        find_slug(norm, features)
    else:
        # pre-crop
        cropped = cropping.correct_rotation(im, crop_props, pad=False, border_erode=parameters.BORDER_ERODE_CZ,
                                            fix_chamfer=False)
        features['bl_uncropped_u8'] = crop_props['mask']
        features['bl_cropped_u8'] = crop_props['mask']
        features['center_y'] = crop_props['center'][0]
        features['center_x'] = crop_props['center'][1]
        features['radius'] = crop_props['radius']
        features['corners'] = crop_props['corners']
        features['center'] = crop_props['center']
        features['crop_rotation'] = 0

        if False:
            view = ImageViewer(im)
            ImageViewer(cropped)
            ImageViewer(crop_props['mask'])
            view.show()

        im = np.ascontiguousarray(cropped, dtype=im.dtype)
        norm = im / hist_features['hist_percentile_99.9']

    # set corners (note: this is for consistency. in current implementation there is no cropping)
    features['corner_tl_x'] = 0
    features['corner_tl_y'] = 0
    features['corner_tr_x'] = w - 1
    features['corner_tr_y'] = 0
    features['corner_br_x'] = w - 1
    features['corner_br_y'] = h - 1
    features['corner_bl_x'] = 0
    features['corner_bl_y'] = h - 1

    if False:
        view = ImageViewer(norm)
        ImageViewer(features['bl_uncropped_u8'])
        view.show()

    if skip_features or ('input_param_skip_features' in features and int(features['input_param_skip_features']) == 1):
        return

    # PL metrics
    hist = ip.histogram_percentiles(im, features, features['center_y'], features['center_x'],
                                    features['radius'])
    if False:
        # features['radius'] = features['radius']
        rgb = create_overlay(im, features)
        # ImageViewer(im)
        ImageViewer(rgb)
        plt.figure()
        plt.plot(hist)
        plt.show()

    # rds
    rds(norm, features)

    # dark/bright corners
    radial_profile(norm, features)

    # rings
    ring_strength(norm, features)


def fill_corners(im, features, edge, dist):
    h, w = im.shape
    if 'radius' in features:
        radius = int(round(features['radius']))
        y2 = int(round(features['center_y']))
        x2 = int(round(features['center_x']))
    elif 'wafer_radius' in features:
        radius = int(round(features['wafer_radius']))
        y2 = int(round(features['wafer_middle_y']))
        x2 = int(round(features['wafer_middle_x']))
    else:
        print "ERROR: No radius found"
        assert False

    h2 = h // 2
    w2 = w // 2

    # pixels to sample intensities along corners
    ys, xs = draw.circle_perimeter(y2, x2, radius - edge)
    mask = ((ys >= 0) & (ys < h) & (xs >= 0) & (xs < w))
    ys = ys[mask]
    xs = xs[mask]
    corner_filled = im.copy()
    corner_avg = 0

    if False:
        im[ys, xs] = im.max() * 1.1
        view = ImageViewer(im)
        view.show()

    # top left
    mask = ((ys < h2) & (xs < w2))
    if mask.sum() > 0:
        corner_val = im[ys[mask], xs[mask]].mean()
        corner_avg += corner_val
        corner_filled[:h2, :w2][dist[:h2, :w2] > radius - edge] = corner_val

    # top right
    mask = ((ys < h2) & (xs > w2))
    if mask.sum() > 0:
        corner_val = im[ys[mask], xs[mask]].mean()
        corner_avg += corner_val
        corner_filled[:h2, w2:][dist[:h2, w2:] > radius - edge] = corner_val

    # bottom left
    mask = ((ys > h2) & (xs < w2))
    if mask.sum() > 0:
        corner_val = im[ys[mask], xs[mask]].mean()
        corner_avg += corner_val
        corner_filled[h2:, :w2][dist[h2:, :w2] > radius - edge] = corner_val

    # bottom right
    mask = ((ys > h2) & (xs > w2))
    if mask.sum() > 0:
        corner_val = im[ys[mask], xs[mask]].mean()
        corner_avg += corner_val
        corner_filled[h2:, w2:][dist[h2:, w2:] > radius - edge] = corner_val

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


# @profile
def ring_strength(im, features):
    DEBUG = False

    # remove a lot of the defects by taking the max of a few positions at equal distance
    h, w = im.shape
    if 'im_center_dist_rot' in features:
        # being called by wafers alg
        dist = features['im_center_dist_rot']
        theta = features['im_center_theta_rot']
        center_x = int(round(features['wafer_middle_x']))
        center_y = int(round(features['wafer_middle_y']))
        radius = int(features['wafer_radius'] - 10)
    else:
        dist, theta = np.empty_like(im, np.float32), np.empty_like(im, np.float32)
        pixel_ops.CenterDistance(dist, theta, features['center_y'], features['center_x'])
        center_x = int(round(features['center_x']))
        center_y = int(round(features['center_y']))
        radius = int(features['radius'] - 10)

    corner_filled, corner_avg = fill_corners(im, features, 10, dist)

    if False:
        view = ImageViewer(im)
        ImageViewer(corner_filled)
        view.show()
        sys.exit()

    maxes = corner_filled.copy()
    rotated = np.empty_like(im)
    for r in [-4.0, -2.0, 2.0, 4.0]:
        rot_mat = cv2.getRotationMatrix2D((center_x, center_y), r, 1.0)
        cv2.warpAffine(corner_filled, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, dst=rotated, borderValue=0)
        maxes = np.maximum(maxes, rotated)

    if False:
        view = ImageViewer(im)
        ImageViewer(maxes)
        view.show()
        sys.exit()

    # A spiral smooth
    # - get coordinates that start at the middle and rotate outwards
    dist = np.round(dist).astype(np.int32)
    dist_flat = dist.flat
    theta_flat = theta.flat

    # first sort by distance from center
    args = np.argsort(dist_flat)
    dist_flat = dist_flat[args]
    theta_flat = theta_flat[args]

    # for pixels at an equal distance, sort by theta
    boundaires = np.where((dist_flat - np.roll(dist_flat, 1)) > 0)[0]
    for i in range(len(boundaires) - 1):
        start = boundaires[i]
        stop = boundaires[i + 1]
        args_t = np.argsort(theta_flat[start:stop])

        args[start:stop] = args[start:stop][args_t]

    # apply smoothing to flattened, ordered image
    im1D = maxes.flatten()
    im1D = im1D[args]
    if False:
        im_smooth = ndimage.gaussian_filter1d(im1D, sigma=30)
    else:
        # faster: smooth downsized
        im_smooth = ndimage.gaussian_filter1d(im1D[::3], sigma=10)
        zoom = len(im1D) / float(len(im_smooth))
        im_smooth = ndimage.zoom(im_smooth, zoom=zoom, order=0)
        assert len(im_smooth) == len(im1D)

    im_rings = im_smooth[np.argsort(args)].reshape((h, w))
    if False:
        im_rings, _ = cz_wafer.fill_corners_edges(im_rings, features, 4, corner_fill=corner_avg)

    if False:
        view = ImageViewer(im)
        ImageViewer(im_rings)
        view.show()
        sys.exit()

    if DEBUG:
        plt.figure()

    rotations = range(0, 361, 10)
    dip_profiles = np.zeros((len(rotations), radius), np.float32)
    circle_strengths = []
    for e, r in enumerate(rotations):
        ys, xs = draw.line(center_y, center_x,
                           center_y + int(radius * math.cos(math.radians(r))),
                           center_x + int(radius * math.sin(math.radians(r))))
        mask = ((ys >= 0) & (xs >= 0) & (ys < h) & (xs < w))
        ys, xs = ys[mask], xs[mask]
        if DEBUG:
            im[ys, xs] = 0

        profile = im_rings[ys, xs]
        sample_r = dist[ys[-1], xs[-1]]

        # resample to standard length
        rs = np.linspace(0, sample_r, num=len(profile), endpoint=True)
        f = interpolate.interp1d(rs, profile)
        profile = f(np.arange(sample_r))

        if parameters.RING_SIGMA1 > 0:
            profile = ndimage.gaussian_filter1d(profile, sigma=parameters.RING_SIGMA1)
        if parameters.RING_SIGMA2 > 0:
            profile_upper = ndimage.gaussian_filter1d(profile, sigma=parameters.RING_SIGMA2)

        # interpolate peaks
        peaks = np.where((profile_upper > np.roll(profile_upper, 1)) &
                         (profile_upper > np.roll(profile_upper, -1)))[0]
        if len(peaks) < 2:
            dip_profiles[e, :len(profile)] = 0
        else:
            f = interpolate.interp1d(peaks, profile_upper[peaks])
            xs = np.arange(peaks[0], peaks[-1])
            f_upper = profile.copy()
            f_upper[xs] = f(xs)

            # find dips
            dip_shape = f_upper - profile

            # ignore middle (small artifacts near middle have disproportionally high radius)
            dip_shape[:100] = 0
            dip_profiles[e, :len(profile)] = dip_shape

            # a second strategy for telling difference between slugs with 1 small dark
            #  and lots/large rings
            zeros = np.where(dip_shape == 0)[0]
            gaps = np.where(zeros[1:] - zeros[:-1] > 1)[0]
            big_dips = []
            for g in gaps:
                start, stop = zeros[g], zeros[g + 1]
                dip_strength = 1000.0 * dip_shape[start:stop].sum() / float(len(profile))
                if dip_strength > 0.5:
                    big_dips.append(dip_strength)
            circle_strengths.append(np.array(big_dips).sum())

        if DEBUG:
            plt.plot(profile)
            plt.plot(dip_profiles[e, :])
            plt.plot(f_upper, '--')

    path_xs = np.zeros(dip_profiles.shape[0], np.int32)
    path_strength = np.zeros_like(dip_profiles, np.float32)
    pixel_ops.strongest_path(dip_profiles, path_strength, path_xs, 15)
    path_vals = dip_profiles[np.arange(dip_profiles.shape[0]), path_xs]

    if False:
        dip_profiles[np.arange(dip_profiles.shape[0]), path_xs] = dip_profiles.max() * 1.1
        view = ImageViewer(dip_profiles)
        ImageViewer(path_strength)
        plt.figure()
        plt.plot(path_strength[-1, :])
        plt.figure()
        plt.plot(path_vals)
        view.show()
        sys.exit()

    # a path might have a few peaks due to non-ring artifacts.
    # - ignore some of the highest areas
    path2 = path_vals.copy()
    for i in range(parameters.NUM_PEAKS):
        m = np.argmax(path2)
        path2[max(0, m - 2):min(path2.shape[0], m + 3)] = 0
    path2[[0, -1]] = 0
    # plt.figure()
    # plt.plot(path_vals)
    # plt.plot(path2)
    # plt.show()

    features['circle_strength'] = 100 * path2.max()
    features['circle_strength_2'] = np.median(circle_strengths) * 10
    # print features['circle_strength_2']

    if DEBUG:
        # plt.plot(dip_profiles.sum(axis=0))
        print features['circle_strength']
        ImageViewer(im)
        ImageViewer(im_rings)
        dip_profiles[np.arange(dip_profiles.shape[0]), path_xs] = dip_profiles.max() * 1.1
        ImageViewer(dip_profiles)
        plt.figure()
        plt.plot(path_vals)
        plt.plot(path2)
        plt.show()

    return im_rings


def main():
   pass


if __name__ == "__main__":
    main()
