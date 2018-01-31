import image_processing as ip
import cv2
import numpy as np
import pixel_ops
from skimage.transform import hough_line, hough_line_peaks  # , probabilistic_hough, probabilistic_hough_line
from skimage import draw
import math
from scipy import ndimage
import matplotlib.pylab as plt
from matplotlib import cm


def resolution(im, features):
    f = {}
    ip.histogram_percentiles(im, f)
    im /= f['hist_percentile_99.9']
    im_smooth = cv2.GaussianBlur(im, ksize=(0, 0), sigmaX=1, borderType=cv2.BORDER_REPLICATE)

    # find peaks
    circle_r = 3
    footprint = np.zeros((circle_r * 2 + 1, circle_r * 2 + 1), np.uint8)
    ys, xs = pixel_ops.circle_perimeter(3, 3, 3)
    footprint[ys, xs] = 1
    ring_maxes = ndimage.maximum_filter(im, footprint=footprint)
    peak_strength = im - ring_maxes

    peaks = np.zeros_like(im, np.uint8)
    pixel_ops.LocalMaxs(im_smooth, peaks)

    marker_locs = np.where((peak_strength > 0.1) & (peaks == 1))
    marker_im = np.zeros_like(peaks)
    marker_im[marker_locs[0], marker_locs[1]] = 1
    struct = ndimage.generate_binary_structure(2, 1)
    marker_im = ndimage.binary_dilation(marker_im, structure=struct)

    # hough transform to find lines
    min_degree = math.asin(1 / float(im.shape[1]))
    increments = math.pi / min_degree
    angles = np.linspace(-np.pi / 2.0, np.pi / 2.0, increments)
    h, theta, d = hough_line(marker_im, angles)

    # find top line candidates
    lines = []
    hspace, angles, dists = hough_line_peaks(h, theta, d, min_distance=5, min_angle=5)
    for i in range(min(6, len(angles))):
        angle = angles[i]
        dist = dists[i]
        y0 = dist / np.sin(angle)
        y1 = (dist - im.shape[1] * np.cos(angle)) / np.sin(angle)
        x0 = 0
        x1 = im.shape[1] - 1
        lines.append((y0, y1, x0, x1))

    if False:
        plt.figure()
        plt.imshow(marker_im, cmap=cm.gray)
        for (y0, y1, x0, x1) in lines:
            plt.plot((x0, x1), (y0, y1), '-r')
        plt.show()

    # sort by vertical position, and discard top and bottom
    lines.sort(key=lambda x: x[0])
    lines = lines[2:-2]

    # pick strongest (interects most peaks)
    strenghs = []
    for (y0, y1, x0, x1) in lines:
        ys, xs = draw.line(int(round(y0)), x0, int(round(y1)), x1)
        y_mask = ((ys >= 0) & (ys < im.shape[0]))
        ys, xs = ys[y_mask], xs[y_mask]
        strenghs.append(marker_im[ys, xs].sum())
    (y0, y1, x0, x1) = lines[np.argmax(strenghs)]
    line_length = math.sqrt(im.shape[1] ** 2 + (y1 - y0) ** 2)
    # compute distance between pixels (> 1 for non horizontal/vertical lines)
    inter_pixel_dist = line_length / im.shape[1]
    ys, xs = draw.line(int(round(y0)), x0, int(round(y1)), x1)
    y_mask = ((ys >= 0) & (ys < im.shape[0]))
    ys, xs = ys[y_mask], xs[y_mask]
    profile = im[ys, xs]

    if False:
        print inter_pixel_dist
        plt.figure()
        plt.imshow(im, cmap=cm.gray)
        plt.plot((x0, x1), (y0, y1), '-r')
        plt.figure()
        plt.plot(profile)
        plt.show()

    # find average dist between profile peaks
    profile_peaks = np.where((profile > np.roll(profile, 1)) &
                             (profile > np.roll(profile, -1)))[0]
    profile_peaks = profile_peaks[2:-2]
    peaks_dists = profile_peaks[1:] - profile_peaks[:-1]

    # remove outliers (due to missed peak or spurious peak)
    median_dist = np.median(peaks_dists)
    peaks_dists = peaks_dists[((peaks_dists >= median_dist - 1) &
                               (peaks_dists <= median_dist + 1))]

    pixels_per_mm = peaks_dists.mean()
    pixels_per_mm *= inter_pixel_dist

    if False:
        print peaks_dists
        print pixels_per_mm
        plt.figure()
        plt.plot(profile)
        plt.plot(profile_peaks, profile[profile_peaks], 'o')
        plt.show()

    features['pixels_per_mm'] = pixels_per_mm


def main():
    pass


if __name__ == '__main__':
    main()
