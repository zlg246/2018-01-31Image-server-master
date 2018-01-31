import numpy as np
import matplotlib.pylab as plt
from image_processing import ImageViewer
import pixel_ops
import cv2
import sys
from scipy import stats, ndimage, optimize, interpolate
import math
import image_processing as ip
import glob
import os
import pandas as pd
plt.style.use('ggplot')


EDGE_THRESH = 0.8


def find_edge(im):
    h, w = im.shape
    edges = cv2.Sobel(im, cv2.CV_32F, 0, 1)*-1

    if False:
        view = ImageViewer(im)
        ImageViewer(edges)
        view.show()

    # find points with strongest gradient
    ys = []
    xs = []
    step_size = w // 100
    for x in range(0, w, step_size):
        col = edges[:, x]
        max_pos = np.argmax(col)
        if col[max_pos] < EDGE_THRESH: continue

        # interpolate position of peak
        # cubic:     0.3320
        # quadratic: 0.3337
        start = max_pos-3
        stop = max_pos+4
        if start < 0 or stop >= h: continue
        r = range(start, stop, 1)
        y = col[start:stop] * -1 # negative because optimization function looks for minimum
        f_cubic = interpolate.interp1d(r, y, kind='cubic', bounds_error=False, fill_value=0)
        peak = optimize.fmin(f_cubic, max_pos, disp=False)[0]

        xs.append(x)
        ys.append(peak)

        if False:
            f_linear = interpolate.interp1d(r, y)
            x_interp = np.linspace(start, stop-1, num=100, endpoint=True)
            plt.figure()
            plt.plot(r, y, 'o', x_interp, f_linear(x_interp), '-', x_interp, f_cubic(x_interp), '--')
            plt.plot(peak, f_cubic(peak), 'ko')
            plt.show()


    POINTS_REQUIRED = 0.5
    points = np.c_[np.array(xs, np.float32).reshape((len(xs), 1)),
                   np.array(ys, np.float32).reshape((len(ys), 1))]

    if False:
        print "Edge points found: %d/%d" % (len(xs), int(POINTS_REQUIRED * (w / float(step_size))))
        plt.figure()
        plt.imshow(im, cmap="gray")
        plt.plot(xs, ys, 'o')
        plt.show()

    if len(xs) < POINTS_REQUIRED * (w / float(step_size)):
        print "Edge points found: %d/%d" % (len(xs), int(POINTS_REQUIRED * (w / float(step_size))))
        sys.exit()

    # vote on best line (to nearest pixel)
    q = 500
    acc = np.zeros((q*2 + 1, q*2 + 1), np.int32)
    pixel_ops.line_vote(points, acc, q, w, 50)
    best_left, best_right = ndimage.maximum_position(acc)
    best_left -= q
    best_right -= q

    if False:
        print best_left, best_right
        view = ImageViewer(acc)
        view.show()
        #sys.exit()

    # exclude any points more than X pixels away
    slope = (best_right - best_left) / float(w)
    ys_fit = slope * points[:, 0] + best_left
    diffs = np.abs(ys_fit - points[:, 1])
    outliers = diffs > 1.5

    # fine tune using least squares
    fit = stats.linregress(points[:, 0][~outliers], points[:, 1][~outliers])
    slope = fit[0]
    intercept = fit[1]
    best_left = intercept
    best_right = slope*(w-1) + intercept

    if False:
        print best_left, best_right
        plt.figure()
        plt.imshow(im, cmap="gray", interpolation="nearest")
        plt.plot(points[:, 0][~outliers], points[:, 1][~outliers], 'go')
        plt.plot(points[:, 0][outliers], points[:, 1][outliers], 'ro')
        plt.plot((0, w-1), (best_left, best_right), '-')
        plt.show()
        #sys.exit()

    return best_left, best_right, points[outliers, :]


def lines_intersect(a1x, a1y, a2x, a2y, b1x, b1y, b2x, b2y):
    if a1x != a2x:
        slopeA = (a2y - a1y) / float(a2x - a1x)
        interceptA = a1y - (slopeA * a1x)
    else:
        slopeA = None

    if b1x != b2x:
        slopeB = (b2y - b1y) / float(b2x - b1x)
        interceptB = b1y - (slopeB * b1x)
    else:
        slopeB = None

    if slopeA is None and slopeB is None:
        print 'Two vertical lines'
        assert False
    elif slopeA is None:
        x = a1x
        y = slopeB * x + interceptB
    elif slopeB is None:
        x = b1x
        y = slopeA * x + interceptA
    else:
        x = (interceptB - interceptA) / float(slopeA - slopeB)
        y = slopeA * x + interceptA

    return y, x


def analyze_pattern(im_orig, output=False, display_image=False):
    features = {}

    """
    sigma tuning:
      0.5 - 0.3327
      1.0 - 0.3320
      1.5 - 0.3343
    """

    # normalisation
    f = {}
    ip.HistogramPercentiles(im_orig, f)
    im = im_orig / f['hist_percentile_95']
    h, w = im.shape

    # smooth
    im_smooth = cv2.GaussianBlur(im, (0, 0), 1.0)

    if False:
        view = ImageViewer(im_orig)
        ImageViewer(im)
        ImageViewer(im_smooth)
        view.show()
        sys.exit()

    # make sure there is something there
    # - check the percentage with intensity < 0.5
    dark = 100.0 * pixel_ops.CountInRange_F32(im_smooth, 0.0, 0.3) / float(h*w)
    if dark > 50:
        print "ERROR: Dark area is %0.02f%%. Is there is a test pattern present?" % (dark)
        sys.exit()

    ##############
    # FIND EDGES #
    ##############
    # Top
    top = im_smooth[:h//4, :]
    tl, tr, outliers1 = find_edge(top)

    # Bottom
    bottom = im_smooth[-h//5:, :][::-1, :]
    bl, br, outliers2 = find_edge(bottom)
    bl = h-bl-1
    br = h-br-1
    outliers2[:, 1] = h-outliers2[:, 1]-1

    # Left
    left = im_smooth[:, :h//5].T
    lt, lb, outliers3 = find_edge(left)
    temp = outliers3[:, 0].copy()
    outliers3[:, 0] = outliers3[:, 1]
    outliers3[:, 1] = temp

    # Right
    right = im_smooth[:, -h//5:][:, ::-1].T
    rt, rb, outliers4 = find_edge(right)
    rt = w-rt-1
    rb = w-rb-1
    temp = outliers4[:, 0].copy()
    outliers4[:, 0] = outliers4[:, 1]
    outliers4[:, 1] = temp
    outliers4[:, 0] = w-outliers4[:, 0]-1

    ################
    # FIND CORNERS #
    ################

    # top left corner
    tlY, tlX = lines_intersect(0, tl, w-1, tr, lt, 0, lb, h-1)
    # top right
    trY, trX = lines_intersect(0, tl, w-1, tr, rt, 0, rb, h-1)
    # bottom left
    blY, blX = lines_intersect(0, bl, w-1, br, lt, 0, lb, h-1)
    # bottom right
    brY, brX = lines_intersect(0, bl, w-1, br, rt, 0, rb, h-1)

    features['length_top'] = math.sqrt((tlY - trY)**2 + (tlX - trX)**2)
    features['length_left'] = math.sqrt((tlY - blY)**2 + (tlX - blX)**2)
    features['length_right'] = math.sqrt((trY - brY)**2 + (trX - brX)**2)
    features['length_bottom'] = math.sqrt((blY - brY)**2 + (blX - brX)**2)

    angle_t = math.degrees(np.arctan2(tl-tr, w))
    angle_l = math.degrees(np.arctan2(h, lt-lb))
    angle_r = math.degrees(np.arctan2(h, rt-rb))
    angle_b = math.degrees(np.arctan2(bl-br, w))

    features['corner_tl'] = 180.0 - (angle_l - angle_t)
    features['corner_tr'] = (angle_r - angle_t)
    features['corner_bl'] = (angle_l - angle_b)
    features['corner_br'] = 180.0 - (angle_r - angle_b)

    corner_vals = np.array([features['corner_tl'], features['corner_tr'],
                            features['corner_bl'], features['corner_br']])
    features['std_corners'] = np.abs(corner_vals - 90).mean()
    if features['corner_tl'] + features['corner_br'] > features['corner_tr'] + features['corner_bl']:
        corner_sign = '+'
    else:
        corner_sign = '-'

    # if max(features['corner_tl'], features['corner_tr'],
    #        features['corner_bl'], features['corner_br']) > 90.5:
    #     print "WARNING: Large angle: %0.03f" % (max(features['corner_tl'], features['corner_tr'],
    #                                                 features['corner_bl'], features['corner_br']))

    if output:
        print """
Edge lengths:
    Top:    %0.02f
    Right:  %0.02f
    Bottom: %0.02f
    Left:   %0.02f

Corner angles:
    Top left:     %0.02f
    Top right:    %0.02f
    Bottom left:  %0.02f
    Bottom right: %0.02f

Mean corner angle deviation from 90: %s%0.02f
""" % (features['length_top'], features['length_right'], features['length_bottom'], features['length_left'],
       features['corner_tl'], features['corner_tr'], features['corner_bl'], features['corner_br'],
       corner_sign, features['std_corners'])

    if display_image:
        plt.figure()
        plt.imshow(im, cmap="gray", interpolation="nearest")
        plt.plot((0, w-1), (tl, tr), 'b-')
        plt.plot((0, w-1), (bl, br), 'b-')
        plt.plot((lt, lb), (0, h-1), 'b-')
        plt.plot((rt, rb), (0, h-1), 'b-')
        plt.plot((tlX, trX, blX, brX), (tlY, trY, blY, brY), 'go')
        plt.xlim((0, w-1))
        plt.ylim((h-1, 0))
        plt.tight_layout()
        plt.show()

    return features


def analyze_folder(folder):
    fns = glob.glob(os.path.join(folder, "*.tif*"))
    fns.sort()
    results = []
    for e, fn in enumerate(fns):
        fn_root = os.path.splitext(os.path.split(fn)[1])[0]
        rotation = fn_root.split('_')[-1]
        if rotation[0] == 'L': rotation = -1*int(rotation[1:])
        elif rotation[0] == 'R': rotation = int(rotation[1:])
        elif rotation[0] == '0': rotation = 0
        else:
            print "Rotation information missing: %s"%(fn)
            rotation = e
        print "Analyze: %s  Rotation: %d" % (fn_root, rotation)

        im = ip.OpenImage(fn).astype(np.float32)
        features = analyze_pattern(im, output=False)

        features['rotation'] = rotation
        #corner_vals = np.array([features[k] for k in features.keys() if k.startswith('corner')])
        #features['corners_std'] = np.abs(corner_vals-90).mean()

        results.append(features)

    df = pd.DataFrame(results)
    df.sort(columns='rotation', inplace=True)
    rotations = df['rotation'].values

    # Corner angles
    corners_vals = df[[k for k in df.columns if k.startswith('corner')]].values.T
    S = 100
    rs = np.linspace(rotations[0], rotations[-1], num=S)
    corners_interpolated = np.zeros((4, S), np.float32)
    for i in range(4):
        f = interpolate.interp1d(rotations, corners_vals[i, :])
        corners_interpolated[i, :] = f(rs)
    corners_std = corners_interpolated.std(axis=0)
    r_op = rs[np.argmin(corners_std)]
    plt.figure()
    plt.plot(rs, corners_std)
    plt.vlines(r_op, 0, corners_std.max(), linestyles='--')
    plt.xlabel("Rotation")
    plt.ylabel("Corner angle standard deviation")
    plt.title("Optimal rotation: %0.02f"%(r_op))


    plt.figure()
    ax = plt.subplot(111)
    for k in df.columns:
        if not k.startswith('corner_'): continue
        if k == 'corner_tl': label = "Top left"
        elif k == 'corner_tr': label = "Top right"
        elif k == 'corner_bl': label = "Bottom left"
        elif k == 'corner_br': label = "Bottom right"
        ax.plot(df['rotation'], df[k], label=label)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.set_xlabel("Rotation")
    ax.set_ylabel("Degrees")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)

    # edge lengths
    plt.figure()
    ax = plt.subplot(111)
    for k in df.columns:
        if not k.startswith('length_'): continue
        ax.plot(df['rotation'], df[k], label=k[len('length_'):])
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
    ax.set_xlabel("Rotation")
    ax.set_ylabel("Pixels")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
              fancybox=True, shadow=True, ncol=5)
    plt.show()


def main():
    if False:
        # single image
        fn = r"C:\Users\Neil\Dropbox (Personal)\BT\Data\tool calibrate\test1.tif"
        im = ip.OpenImage(fn).astype(np.float32)
        if False:
            im = np.ascontiguousarray(np.rot90(im))
        analyze_pattern(im, output=True)
    elif False:
        # set
        folder = r"C:\Users\Neil\Dropbox (Personal)\BT\Data\tool calibrate\2016-04-21"
        analyze_folder(folder)
    else:
        if len(sys.argv) < 2:
            print "Please input a file or folder as input"
            sys.exit()
        if os.path.isfile(sys.argv[1]):
            fn = sys.argv[1]
            im = ip.OpenImage(fn).astype(np.float32)
            analyze_pattern(im, output=True)
        elif os.path.isdir(sys.argv[1]):
            folder = sys.argv[1]
            print folder
            analyze_folder(folder)


if __name__ == '__main__':
    main()
