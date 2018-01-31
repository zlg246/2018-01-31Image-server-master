r"""Cython functions to do low level operations

Sample build:

C:\Users\Neil\Miniconda2\envs\BT\python.exe compile_pixel_ops.py build_ext --inplace  --compiler=msvc

profile:

C:\Users\Neil\Miniconda2\envs\BT\python.exe C:\Users\Neil\Anaconda\Lib\site-packages\cython.py -a pixel_ops.pyx
"""
import sys
import numpy as np
cimport numpy as np
cimport cython

cdef extern from *:
    ctypedef void const_void "const void"


from libc.math cimport sin
from libc.math cimport cos
from libc.math cimport sqrt
from libc.math cimport atan2
from libc.math cimport abs

cdef int cmp(const_void *a, const_void *b):
    cdef float v = (<float*> a)[0] - (<float*> b)[0]
    if v < 0: return -1
    if v > 0: return 1
    return 0

cdef inline int round(double x):
    if x < 0.0:
        return <int> (x - 0.5)
    else:
        return <int> (x + 0.5)

cdef inline int floor(double x):
    return <int> (x - (x % 1))

@cython.boundscheck(False)
@cython.wraparound(False)
def UpdateMask(np.ndarray[dtype=np.uint16_t, ndim=2, negative_indices=False, mode='c'] orig_mask,
               np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result_mask,
               unsigned short orig_label, unsigned char new_label):
    cdef:
        int w, h, y, x

    h = orig_mask.shape[0]
    w = orig_mask.shape[1]
    for y in range(h):
        for x in range(w):
            if orig_mask[y, x] == orig_label: result_mask[y, x] = new_label

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def UpdateMaskU32U8(np.ndarray[dtype=np.int32_t, ndim=2, negative_indices=False, mode='c'] orig_mask,
                    np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result_mask,
                    int orig_label, unsigned char new_label):
    cdef:
        int w, h, y, x

    h = orig_mask.shape[0]
    w = orig_mask.shape[1]
    for y in range(h):
        for x in range(w):
            if orig_mask[y, x] == orig_label: result_mask[y, x] = new_label

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def WaferFit(params, float center_y, float center_x,
             np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False, mode='c'] ys,
             np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False, mode='c'] xs,
             np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] edges,
             int update,
             int im_height,
             int im_width):
    cdef:
        int i, x, y
        float shift_y, shift_x
        float theta, scale, edge_total, count
        int num_coords
        float y_c, x_c
        float y_r, x_r, avg

    (theta, shift_y, shift_x, scale) = params

    num_coords = ys.shape[0]
    edge_total = 0
    count = 0
    for i in range(num_coords):
        y_c = (ys[i] - center_y) * scale
        x_c = (xs[i] - center_x) * scale

        y_r = ((x_c * sin(theta)) + (y_c * cos(theta)))
        x_r = ((x_c * cos(theta)) - (y_c * sin(theta)))

        y_r += (center_y + shift_y)
        x_r += (center_x + shift_x)

        x = <int> round(x_r)
        y = <int> round(y_r)

        if y >= 0 and y < im_height and x >= 0 and x < im_width:
            edge_total += edges[y, x]
            count += 1

            if update == 1: edges[y, x] = 0

    avg = 0
    if count >= 1:
        avg = -(edge_total / count)

    return avg

@cython.boundscheck(False)
@cython.wraparound(False)
def strongest_path(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im,
                   np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] path_strength,
                   np.ndarray[dtype=np.int32_t, ndim=1, negative_indices=False, mode='c'] path_xs,
                   int k):
    cdef:
        int w, h, y, x, c, max_pos, max_pos1, max_pos2
        float max_val

    h = im.shape[0]
    w = im.shape[1]

    # copy first row
    for x in range(w):
        path_strength[0, x] = im[0, x]

    # sum down
    for y in range(1, h):
        # max filter to row above
        for c in range(w):
            max_val = 0
            for x in range(max(0, c - k), min(w, c + k)):
                if path_strength[y - 1, x] > max_val:
                    max_val = path_strength[y - 1, x]
            path_strength[y, c] = im[y, c] + max_val

    # work way back up
    max_val = 0
    max_pos1 = -1
    max_pos2 = -1
    for x in range(w):
        if path_strength[h - 1, x] > max_val:
            max_val = path_strength[h - 1, x]
            max_pos1 = x
            max_pos2 = -1
        elif path_strength[h - 1, x] == max_val:
            max_pos2 = x
    if max_pos2 == -1:
        path_xs[h - 1] = max_pos1
    else:
        path_xs[h - 1] = (max_pos1 + max_pos2) // 2

    for y in range(h - 2, -1, -1):
        c = path_xs[y + 1]
        max_val = 0
        for x in range(max(0, c - k), min(w, c + k)):
            if path_strength[y, x] > max_val:
                max_val = path_strength[y, x]
                max_pos = x
        if im[y, max_pos] > 0.01:
            path_xs[y] = max_pos
        else:
            path_xs[y] = c

@cython.boundscheck(False)
@cython.wraparound(False)
def UpdateMaskI32(np.ndarray[dtype=np.int32_t, ndim=2, negative_indices=False, mode='c'] orig_mask,
                  np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result_mask,
                  unsigned short orig_label, unsigned char new_label):
    cdef:
        int w, h, y, x

    h = orig_mask.shape[0]
    w = orig_mask.shape[1]
    for y in range(h):
        for x in range(w):
            if orig_mask[y, x] == orig_label: result_mask[y, x] = new_label

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def UpdateMaskF32(np.ndarray[dtype=np.uint16_t, ndim=2, negative_indices=False, mode='c'] orig_mask,
                  np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] result_mask,
                  unsigned short orig_label, float new_label):
    cdef:
        int w, h, y, x

    h = orig_mask.shape[0]
    w = orig_mask.shape[1]
    for y in range(h):
        for x in range(w):
            if orig_mask[y, x] == orig_label: result_mask[y, x] = new_label

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CopyMaskF32(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig_vals,
                np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] result_vals,
                np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
                unsigned short mask_label):
    cdef:
        int w, h, y, x

    h = orig_vals.shape[0]
    w = orig_vals.shape[1]
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_label: result_vals[y, x] = orig_vals[y, x]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def GetMaskValues(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                  np.ndarray[dtype=np.int32_t, ndim=2, negative_indices=False, mode='c'] mask,
                  np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False, mode='c'] values,
                  int mask_value):
    cdef:
        int w, h, y, x, count

    h = orig.shape[0]
    w = orig.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_value:
                values[count] = orig[y, x]
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdLT_F32(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                         np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] result,
                         float threshold, float new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdLT_F64(np.ndarray[dtype=np.float64_t, ndim=2, negative_indices=False, mode='c'] orig,
                         np.ndarray[dtype=np.float64_t, ndim=2, negative_indices=False, mode='c'] result,
                         float threshold, float new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdLT_F32_U8(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                            np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                            float threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdLT_U8_F32(np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] orig,
                            np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] result,
                            unsigned char threshold, float new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdGT_U8_F32(np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] orig,
                            np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] result,
                            unsigned char threshold, float new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdRange_F32_U8(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                               np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                               float lb, float ub, unsigned char new_value):
    cdef:
        int w, h, y, x, count

    count = 0
    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] >= lb and orig[y, x] <= ub:
                result[y, x] = new_value
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdLT_I32_U8(np.ndarray[dtype=np.int32_t, ndim=2, negative_indices=False, mode='c'] orig,
                            np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                            int threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdGT_I32_U8(np.ndarray[dtype=np.int32_t, ndim=2, negative_indices=False, mode='c'] orig,
                            np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                            int threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdLT_U8_U8(np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] orig,
                           np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                           unsigned char threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdGT_U8_U8(np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] orig,
                           np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                           unsigned char threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdGT_F32(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                         np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] result,
                         float threshold, float new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdGT_F64(np.ndarray[dtype=np.float64_t, ndim=2, negative_indices=False, mode='c'] orig,
                         np.ndarray[dtype=np.float64_t, ndim=2, negative_indices=False, mode='c'] result,
                         float threshold, float new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ApplyThresholdGT_F32_U8(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                            np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                            float threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] = new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def AddThresholdGT_F32_U8(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                          np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] result,
                          float threshold, unsigned char new_value):
    cdef:
        int w, h, y, x

    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                result[y, x] += new_value

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CountThresholdGT_F32(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                         float threshold):
    cdef:
        int w, h, y, x, count

    h = orig.shape[0]
    w = orig.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if orig[y, x] > threshold:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountThresholdLT_F32(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                         float threshold):
    cdef:
        int w, h, y, x, count

    h = orig.shape[0]
    w = orig.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if orig[y, x] < threshold:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountInMaskEqual_U8(np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] orig,
                        np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
                        unsigned char mask_value, unsigned char orig_value):
    cdef:
        int w, h, y, x, count

    h = orig.shape[0]
    w = orig.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_value and orig[y, x] == orig_value:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountGT_F32_U8(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                   np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
                   unsigned char mask_value, float orig_value):
    cdef:
        int w, h, y, x, count

    h = orig.shape[0]
    w = orig.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_value and orig[y, x] >= orig_value:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountLT_F32_U8(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] orig,
                   np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
                   unsigned char mask_value, float orig_value):
    cdef:
        int w, h, y, x, count

    h = orig.shape[0]
    w = orig.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_value and orig[y, x] <= orig_value:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountEqual_I32(np.ndarray[dtype=np.int32_t, ndim=2, negative_indices=False, mode='c'] mask,
                   int mask_value):
    cdef:
        int w, h, y, x, count

    h = mask.shape[0]
    w = mask.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_value:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountEqual_U8(np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
                  unsigned char mask_value):
    cdef:
        int w, h, y, x, count

    h = mask.shape[0]
    w = mask.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_value:
                count += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def CountInRange_F32(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im,
                     float low, float high):
    cdef:
        int w, h, y, x, count

    h = im.shape[0]
    w = im.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if im[y, x] >= low and im[y, x] <= high:
                count += 1

    return count

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def FillPolarBackground(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im):
#    cdef:
#        int w, h, y1, y2, x, y
#        float v1, v2, s, steps
#
#    h = im.shape[0]
#    w = im.shape[1]
#    for x in range(w):
#        y1 = -1
#        for y2 in range(1, h):
#            if im[y2, x] > 0:
#                if im[y2-1, x] == 0 and y1 > 0:
#                    # interp y1 to y2
#                    v1 = im[y1, x]
#                    v2 = im[y2, x]
#                    s = 0
#                    steps = y2-y1+1
#                    for y in range(y1, y2+1):
#                        im[y, x] = (s/steps)*v2 + ((steps-s)/steps)*v1
#                        s += 1
#                y1 = y2

#    steps = (2*bb_width) + 1
#    for bb in locations:
#        for y in range(h):
#            v1 = im[y, bb-bb_width]
#            v2 = im[y, bb+bb_width+1]
#            d = v2 - v1
#            s = 0

#            for x in range(bb-bb_width, bb+bb_width+1):
#                im[y, x] = (s/steps)*v2 + ((steps-s)/steps)*v1
#                s += 1


@cython.boundscheck(False)
@cython.wraparound(False)
def BackgrounHistogram(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im,
                       np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False, mode='c'] hist):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        for x in range(w):
            if im[y, x] < 0.2:
                hist[0] += 1
            elif im[y, x] < 0.4:
                hist[1] += 1
            elif im[y, x] < 0.6:
                hist[2] += 1
            elif im[y, x] < 0.8:
                hist[3] += 1
            else:
                hist[4] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def BackgrounHistogramMask(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im,
                           np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] mask,
                           np.ndarray[dtype=np.float32_t, ndim=1, negative_indices=False, mode='c'] hist):
    cdef:
        int w, h, y, x, count

    h = im.shape[0]
    w = im.shape[1]

    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] != 0: continue
            count += 1
            if im[y, x] < 0.2:
                hist[0] += 1
            elif im[y, x] < 0.4:
                hist[1] += 1
            elif im[y, x] < 0.6:
                hist[2] += 1
            elif im[y, x] < 0.8:
                hist[3] += 1
            else:
                hist[4] += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def ClipImage(np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] im,
              float t_lower, float t_upper):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if im[y, x] > t_upper:
                im[y, x] = t_upper
            elif im[y, x] < t_lower:
                im[y, x] = t_lower

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ClipImageF64(np.ndarray[dtype=np.float64_t, ndim=2,
                            negative_indices=False, mode='c'] im,
                 float t_lower, float t_upper):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if im[y, x] > t_upper:
                im[y, x] = t_upper
            elif im[y, x] < t_lower:
                im[y, x] = t_lower

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def InitRectangle(np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] im,
                  int t, int b, int l, int r):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if y < t or y >= b or x < l or x >= r:
                im[y, x] = 0
            else:
                im[y, x] = 1

@cython.boundscheck(False)
@cython.wraparound(False)
def MaskAvgDiff(np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] im,
                np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] mask):
    cdef:
        int w, h, y, x
        float total_in, total_out, count_in, count_out

    total_in, total_out, count_in, count_out = 0, 0, 0, 0
    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if mask[y, x] < 0.5:
                total_out += im[y, x]
                count_out += 1
            else:
                total_in += im[y, x]
                count_in += 1

    return total_in / count_in, total_out / count_out

@cython.boundscheck(False)
@cython.wraparound(False)
def CellChips(np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] contours,
              np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] convex_hull,
              np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] chips):
    cdef:
        int w, h, y, x, chip

    h, w = contours.shape[0], contours.shape[1]
    for y in range(h):
        # left-right
        x = 0
        chip = 0
        while x < w:
            if contours[y, x] == 1:
                break
            if convex_hull[y, x] == 1:
                chip = 1
            elif chip == 1:
                chips[y, x] = 1
            x += 1

        # right-left
        x = w - 1
        chip = 0
        while x > 0:
            if contours[y, x] == 1:
                break
            if convex_hull[y, x] == 1:
                chip = 1
            elif chip == 1:
                chips[y, x] = 1
            x -= 1

    for x in range(w):
        # top-down
        y = 0
        chip = 0
        while y < h:
            if contours[y, x] == 1:
                break
            if convex_hull[y, x] == 1:
                chip = 1
            elif chip == 1:
                chips[y, x] = 1
            y += 1

        # bottom-up
        y = h - 1
        chip = 0
        while y > 0:
            if contours[y, x] == 1:
                break
            if convex_hull[y, x] == 1:
                chip = 1
            elif chip == 1:
                chips[y, x] = 1
            y -= 1

@cython.boundscheck(False)
@cython.wraparound(False)
def CellContour(np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] im,
                np.ndarray[dtype=np.uint8_t, ndim=2,
                           negative_indices=False, mode='c'] contours,
                np.ndarray[dtype=np.uint8_t, ndim=2,
                           negative_indices=False, mode='c'] corners,
                float threshold):
    cdef:
        int w, h, y, x

    # pass 1: find contours
    h, w = im.shape[0], im.shape[1]
    for y in range(h):
        # left-right
        x = 0
        while x < w:
            if im[y, x] >= threshold:
                contours[y, x] = 1
                break
            x += 1

        # right-left
        x = w - 1
        while x > 0:
            if im[y, x] >= threshold:
                contours[y, x] = 1
                break
            x -= 1

    for x in range(w):
        # top-down
        y = 0
        while y < h:
            if im[y, x] >= threshold:
                contours[y, x] = 1
                break
            y += 1

        # right-left
        y = h - 1
        while y > 0:
            if im[y, x] >= threshold:
                contours[y, x] = 1
                break
            y -= 1

    # pass 2: find corner points
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if contours[y, x] == 0: continue

            if ((contours[y - 1, x - 1] == 1 and contours[y + 1, x + 1] == 1) or
                    (contours[y - 1, x] == 1 and contours[y + 1, x] == 1) or
                    (contours[y - 1, x + 1] == 1 and contours[y + 1, x - 1] == 1) or
                    (contours[y, x - 1] == 1 and contours[y, x + 1] == 1)):
                continue

            corners[y, x] = 1

@cython.boundscheck(False)
@cython.wraparound(False)
def FillFiltered(np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] im,
                 np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] filtered,
                 np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] filtered_h,
                 np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] filtered_v,
                 np.ndarray[dtype=np.uint8_t, ndim=2,
                            negative_indices=False, mode='c'] bb_mask,
                 np.ndarray[dtype=np.float32_t, ndim=1,
                            negative_indices=False, mode='c'] bb_right,
                 np.ndarray[dtype=np.float32_t, ndim=1,
                            negative_indices=False, mode='c'] bb_left,
                 int f_len, int radius):
    cdef:
        int w, h, y, x, count_left, count_right, i, s, w2, h2, r2, yd
        float v1, v2

    s = bb_right.shape[0]
    h, w = filtered.shape[0], filtered.shape[1]
    w2 = w // 2
    h2 = h // 2
    r2 = radius * radius
    for y in range(h):
        yd = (y - h2) * (y - h2)
        for x in range(w):
            if yd + ((x - w2) * (x - w2)) > r2:
                filtered[y, x] = im[y, x]
            elif x < f_len or x >= w - f_len or bb_mask[y, x] == 1:
                filtered[y, x] = filtered_v[y, x]
            else:
                filtered[y, x] = filtered_h[y, x]

    if False: return

    # blending
    count_left, count_right = 0, 0
    for y in range(h):
        for x in range(w):
            if bb_mask[y, x] == 1 and bb_mask[y, x + 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x + s]
                #if v1 == v2: continue
                if v1 > v2 or v2 - v1 < 0.01: continue
                for i in range(s):
                    bb_right[i] += (filtered_v[y, x + i] - v1) / float(v2 - v1)
                count_right += 1
            elif bb_mask[y, x] == 1 and bb_mask[y, x - 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x - s]
                #if v1 == v2: continue
                if v1 > v2 or v2 - v1 < 0.01: continue
                for i in range(s): bb_left[i] += (filtered_v[y, x - i] - v1) / float(v2 - v1)
                count_left += 1

    if False: return
    for i in range(s):
        bb_right[i] /= count_right
        bb_left[i] /= count_left

    for y in range(h):
        for x in range(w):
            if bb_mask[y, x] == 1 and bb_mask[y, x + 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x + s]
                if v1 == v2:
                    for i in range(s):
                        filtered[y, x + i] = v1
                else:
                    for i in range(s):
                        filtered[y, x + i] = min(1.0, v1 + bb_right[i] * (v2 - v1))
            elif bb_mask[y, x] == 1 and bb_mask[y, x - 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x - s]
                if v1 == v2:
                    for i in range(s):
                        filtered[y, x - i] = v1
                else:
                    for i in range(s):
                        filtered[y, x - i] = min(1.0, v1 + bb_left[i] * (v2 - v1))

@cython.boundscheck(False)
@cython.wraparound(False)
def FillFilteredOLD(np.ndarray[dtype=np.uint8_t, ndim=2,
                               negative_indices=False, mode='c'] filtered,
                    np.ndarray[dtype=np.uint8_t, ndim=2,
                               negative_indices=False, mode='c'] filtered_h,
                    np.ndarray[dtype=np.uint8_t, ndim=2,
                               negative_indices=False, mode='c'] filtered_v,
                    np.ndarray[dtype=np.uint8_t, ndim=2,
                               negative_indices=False, mode='c'] bb_mask,
                    np.ndarray[dtype=np.float32_t, ndim=1,
                               negative_indices=False, mode='c'] bb_right,
                    np.ndarray[dtype=np.float32_t, ndim=1,
                               negative_indices=False, mode='c'] bb_left,
                    int f_len):
    cdef:
        int w, h, y, x, count_left, count_right, i, s
        int v1, v2

    s = bb_right.shape[0]
    h, w = filtered.shape[0], filtered.shape[1]
    for y in range(h):
        for x in range(w):
            if x < f_len or x >= w - f_len or bb_mask[y, x] == 1:
                filtered[y, x] = filtered_v[y, x]
            else:
                filtered[y, x] = filtered_h[y, x]

    # blending
    count_left, count_right = 0, 0
    for y in range(h):
        for x in range(w):
            if bb_mask[y, x] == 1 and bb_mask[y, x + 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x + s]
                if v1 == v2: continue
                for i in range(s): bb_right[i] += (filtered_v[y, x + i] - v1) / float(v2 - v1)
                count_right += 1
            elif bb_mask[y, x] == 1 and bb_mask[y, x - 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x - s]
                if v1 == v2: continue
                for i in range(s): bb_left[i] += (filtered_v[y, x - i] - v1) / float(v2 - v1)
                count_left += 1

    for i in range(s):
        bb_right[i] /= count_right
        bb_left[i] /= count_left

    for y in range(h):
        for x in range(w):
            if bb_mask[y, x] == 1 and bb_mask[y, x + 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x + s]
                if v1 == v2:
                    for i in range(s):
                        filtered[y, x + i] = v1
                else:
                    for i in range(s):
                        filtered[y, x + i] = min(255, v1 + int(bb_right[i] * (v2 - v1)))
            elif bb_mask[y, x] == 1 and bb_mask[y, x - 1] == 0:
                v1 = filtered_v[y, x]
                v2 = filtered_v[y, x - s]
                if v1 == v2:
                    for i in range(s):
                        filtered[y, x - i] = v1
                else:
                    for i in range(s):
                        filtered[y, x - i] = min(255, v1 + int(bb_left[i] * (v2 - v1)))

@cython.boundscheck(False)
@cython.wraparound(False)
def CopyRows(np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] im_in,
             np.ndarray[dtype=np.int32_t, ndim=1,
                        negative_indices=False, mode='c'] rows,
             np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] im_out):
    cdef:
        int w, h, y, x, n, r
        #float a, b, c, d, e

    h, w = im_in.shape[0], im_in.shape[1]
    n = rows.shape[0]
    for r in range(len(rows)):
        y = rows[r]
        for x in range(w):
            im_out[r, x] = im_in[y, x]

@cython.boundscheck(False)
@cython.wraparound(False)
def FilterH(np.ndarray[dtype=np.float32_t, ndim=2,
                       negative_indices=False, mode='c'] im_in,
            np.ndarray[dtype=np.float32_t, ndim=2,
                       negative_indices=False, mode='c'] im_out,
            int s):
    cdef:
        int w, h, y, x
        float a, b, c, d, e

    h, w = im_in.shape[0], im_in.shape[1]

    for y in range(h):
        for x in range(2 * s, w - (2 * s)):
            a, b, c, d, e = im_in[y, x - (2 * s)], im_in[y, x - s], im_in[y, x], im_in[y, x + s], im_in[y, x + (2 * s)]
            if b < a: a, b = b, a
            if d < c: c, d = d, c
            if c < a:
                b, d = d, b
                c = a
            a = e
            if b < a: b, a = a, b
            if a < c:
                d = b
                a = c

            im_out[y, x] = min(a, d)

@cython.boundscheck(False)
@cython.wraparound(False)
def FilterV(np.ndarray[dtype=np.float32_t, ndim=2,
                       negative_indices=False, mode='c'] im_in,
            np.ndarray[dtype=np.float32_t, ndim=2,
                       negative_indices=False, mode='c'] im_out):
    cdef:
        int w, h, y, x
        float a, b, c, d, e

    h, w = im_in.shape[0], im_in.shape[1]

    for y in range(2, h - 2):
        for x in range(w):
            a, b, c, d, e = im_in[y - 2, x], im_in[y - 1, x], im_in[y, x], im_in[y + 1, x], im_in[y + 2, x]
            if b < a: a, b = b, a
            if d < c: c, d = d, c
            if c < a:
                b, d = d, b
                c = a
            a = e
            if b < a: b, a = a, b
            if a < c:
                d = b
                a = c

            im_out[y, x] = min(a, d)

@cython.boundscheck(False)
@cython.wraparound(False)
def BrightLineBreaks(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] bright_lines,
                     np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] cc_sums,
                     np.ndarray[dtype=np.uint8_t, ndim=2, negative_indices=False, mode='c'] breaks,
                     float bright_threshold, float sum_threshold, int min_length):
    cdef:
        int w, h, y, x, cc_num, in_line, start_pos, max_pos
        float cc_sum, max_val
        #float a, b, c, d, e

    h, w = bright_lines.shape[0], bright_lines.shape[1]

    for y in range(h):
        in_line = 0
        cc_sum = 0
        for x in range(w):
            if bright_lines[y, x] > bright_threshold:
                cc_sum += bright_lines[y, x]
                if in_line == 0:
                    start_pos = x
                    in_line = 1
                    #    max_pos = x
                    #    max_val = bright_lines[y, x]
                    #else:
                    #    if bright_lines[y, x] > max_val:
                    #        max_val = bright_lines[y, x]
                    #        max_pos = x
            elif in_line == 1:
                in_line = 0
                x2 = x - 1
                if cc_sum > sum_threshold and x - start_pos > min_length:
                    #breaks[y, max_pos] = 1
                    if bright_lines[y, x - 3] > bright_lines[y, start_pos + 3]:
                        breaks[y, x] = 1
                    else:
                        breaks[y, start_pos] = 1
                    while bright_lines[y, x2] > bright_threshold:
                        cc_sums[y, x2] = cc_sum
                        x2 -= 1
                cc_sum = 0

@cython.boundscheck(False)
@cython.wraparound(False)
def FillBars(np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] im,
             np.ndarray[dtype=np.uint8_t, ndim=2,
                        negative_indices=False, mode='c'] mask,
             int s):
    cdef:
        int w, h, y, x, i
        float local_sum
        int local_count

    h = im.shape[0]
    w = im.shape[1]
    for y in range(s, h - s - 1):
        for x in range(s, w - s - 1):
            if mask[y, x] == 0: continue
            local_sum = 0
            local_count = 0
            for i in range(-s, s + 1):
                if mask[y, x + i] == 1: continue
                local_sum += im[y, x + i]
                local_count += 1
            if local_count == 0:
                im[y, x] = 0
            else:
                im[y, x] = local_sum / local_count

@cython.boundscheck(False)
@cython.wraparound(False)
def BinaryThreshold(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    float threshold):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if im[y, x] > threshold:
                im[y, x] = 1
            else:
                im[y, x] = 0

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def MaskSum_U8(np.ndarray[dtype=np.uint8_t, ndim=2,
                          negative_indices=False, mode='c'] orig,
               np.ndarray[dtype=np.uint8_t, ndim=2,
                          negative_indices=False, mode='c'] mask,
               unsigned char mask_val):
    cdef:
        int w, h, y, x, total

    total = 0
    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_val:
                total += orig[y, x]

    return total

@cython.boundscheck(False)
@cython.wraparound(False)
def MaskMean_F32(np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] orig,
                 np.ndarray[dtype=np.uint8_t, ndim=2,
                            negative_indices=False, mode='c'] mask,
                 unsigned char mask_val):
    cdef:
        int w, h, y, x,
        float count, total

    total = 0
    count = 0
    h = orig.shape[0]
    w = orig.shape[1]
    for y in range(h):
        for x in range(w):
            if mask[y, x] == mask_val:
                total += orig[y, x]
                count += 1

    if count == 0:
        return 0
    else:
        return total / count

@cython.boundscheck(False)
@cython.wraparound(False)
def FastHistogram(np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] im,
                  np.ndarray[dtype=np.int32_t, ndim=1,
                             negative_indices=False, mode='c'] hist):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if im[y, x] < 0:
                hist[0] += 1
            else:
                hist[<int> im[y, x]] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FastHistogramMask(np.ndarray[dtype=np.float32_t, ndim=2,
                                 negative_indices=False, mode='c'] im,
                      np.ndarray[dtype=np.uint8_t, ndim=2,
                                 negative_indices=False, mode='c'] mask,
                      np.ndarray[dtype=np.int32_t, ndim=1,
                                 negative_indices=False, mode='c'] hist):
    cdef:
        int w, h, y, x, count

    h = im.shape[0]
    w = im.shape[1]
    count = 0
    for y in range(h):
        for x in range(w):
            if mask[y, x] != 0: continue
            count += 1
            hist[<int> im[y, x]] += 1

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def FastHistogram1D(np.ndarray[dtype=np.float32_t, ndim=1,
                               negative_indices=False, mode='c'] im,
                    np.ndarray[dtype=np.int32_t, ndim=1,
                               negative_indices=False, mode='c'] hist):
    cdef:
        int w, x

    w = im.shape[0]
    for x in range(w):
        if im[x] < 0:
            hist[0] += 1
        else:
            hist[<int> im[x]] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def DistCenterThresh(np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] im,
                     float max_dist, int center_x, int center_y, float new_val):
    cdef:
        int x, y, w, h, y2
        float r

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        y2 = (y - center_y) ** 2
        for x in range(w):
            r = sqrt(y2 + (x - center_x) ** 2)
            if r > max_dist: im[y, x] = new_val

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def DistCenterRange(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    float min_dist, float max_dist, int center_x, int center_y, float new_val):
    cdef:
        int x, y, w, h, y2
        float r

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        y2 = (y - center_y) ** 2
        for x in range(w):
            r = sqrt(y2 + (x - center_x) ** 2)
            if r < min_dist or r > max_dist: im[y, x] = new_val

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CenterDistance(np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] r,
                   np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] theta,
                   int h2, int w2):
    cdef:
        int w, h, y, x

    h = r.shape[0]
    w = r.shape[1]
    for y in range(h):
        for x in range(w):
            r[y, x] = sqrt(((y - h2) * (y - h2)) + ((x - w2) * (x - w2)))
            theta[y, x] = atan2(y - h2, x - w2)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CenterDistanceF64(np.ndarray[dtype=np.float64_t, ndim=2,
                                 negative_indices=False, mode='c'] r,
                      np.ndarray[dtype=np.float64_t, ndim=2,
                                 negative_indices=False, mode='c'] theta,
                      int h2, int w2):
    cdef:
        int w, h, y, x

    h = r.shape[0]
    w = r.shape[1]
    for y in range(h):
        for x in range(w):
            r[y, x] = sqrt(((y - h2) * (y - h2)) + ((x - w2) * (x - w2)))
            theta[y, x] = atan2(y - h2, x - w2)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def GetDiags(np.ndarray[dtype=np.uint8_t, ndim=2,
                        negative_indices=False, mode='c'] im,
             int m_y, int m_x):
    cdef:
        int w, h, y, x, c

    ds = []
    h = im.shape[0]
    w = im.shape[1]

    x, y = m_x, m_y
    while x > 0 and y > 0 and im[y, x] == 1: x -= 1; y -= 1
    ds.append((y, x))

    x, y = m_x, m_y
    while x < w and y > 0 and im[y, x] == 1: x += 1; y -= 1
    ds.append((y, x))

    x, y = m_x, m_y
    while x < w and y < h and im[y, x] == 1: x += 1; y += 1
    ds.append((y, x))

    x, y = m_x, m_y
    while x > 0 and y < h and im[y, x] == 1: x -= 1; y += 1
    ds.append((y, x))

    return ds

@cython.boundscheck(False)
@cython.wraparound(False)
def HighlightCrackSP(np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] im,
                     np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] filtered,
                     np.ndarray[dtype=np.int32_t, ndim=2,
                                negative_indices=False, mode='c'] filters,
                     float background_val):
    cdef:
        int w, h, y, x, s
        int num_filters
        float v1, v2, v_or, v_max
        int i_max

    h = im.shape[0]
    w = im.shape[1]
    s = filters.max()
    num_filters = len(filters)

    for y in range(h):
        for x in range(w):
            if (y < s or x < s or y > h - s - 1 or x > w - s - 1):
                filtered[y, x] = background_val
                continue

            v_max = -1
            i_max = -1
            for i in range(num_filters):
                v1 = im[y + filters[i, 0], x + filters[i, 1]]
                v2 = im[y + filters[i, 2], x + filters[i, 3]]

                v_or = (v1 + v2) / 2.0

                if v_or > v_max:
                    v_max = v_or
                    i_max = i

            filtered[y, x] = (im[y, x] -
                              max(im[y + filters[i_max, 4], x + filters[i_max, 5]],
                                  im[y + filters[i_max, 6], x + filters[i_max, 7]]))

            if filtered[y, x] < 0:
                filtered[y, x] = 0

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def StarScracksCZ(np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] im,
                  float thresh,
                  np.ndarray[dtype=np.float32_t, ndim=3,
                             negative_indices=False, mode='c'] filters,
                  np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] filtered):
    cdef:
        int w, h, y, x, i, j, s, s2, r, num_filters
        float fs[6]


    h = im.shape[0]
    w = im.shape[1]
    s = filters.shape[1]
    s2 = (filters.shape[1] - 1) / 2
    num_filters = filters.shape[0]

    for y in range(s2, h - s2):
        for x in range(s2, w - s2):
            if im[y, x] < thresh: continue

            for r in range(num_filters):
                fs[r] = 0

            for r in range(num_filters):
                for j in range(s):
                    for i in range(s):
                        fs[r] += im[y - s2 + j, x - s2 + i] * filters[r, j, i]

            if fs[0] + fs[1] + fs[2] > fs[3] + fs[4] + fs[5]:
                filtered[y, x] = max(fs[3], fs[4], fs[5])
            else:
                filtered[y, x] = max(fs[0], fs[1], fs[2])

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FillBackgroundCZ(np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] im,
                     float val,
                     int edge, float radius,
                     np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] ds):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if (x < edge or y < edge or (w - x) < edge or (h - y) < edge or ds[y, x] > radius):
                im[y, x] = val

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FilterCracksCZ(np.ndarray[dtype=np.int32_t, ndim=2,
                              negative_indices=False, mode='c'] crack_ccs1,
                   np.ndarray[dtype=np.int32_t, ndim=1,
                              negative_indices=False, mode='c'] scratch1,
                   np.ndarray[dtype=np.int32_t, ndim=2,
                              negative_indices=False, mode='c'] crack_ccs2,
                   np.ndarray[dtype=np.int32_t, ndim=1,
                              negative_indices=False, mode='c'] scratch2,
                   np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] cracks,
                   ):
    cdef:
        int w, h, y, x

    h = crack_ccs1.shape[0]
    w = crack_ccs1.shape[1]
    for y in range(h):
        for x in range(w):
            if crack_ccs1[y, x] > 0 and crack_ccs2[y, x] > 0:
                scratch1[crack_ccs1[y, x]] = 1
                scratch2[crack_ccs2[y, x]] = 1

    for y in range(h):
        for x in range(w):
            if scratch1[crack_ccs1[y, x]] == 1 or scratch2[crack_ccs2[y, x]] == 1:
                cracks[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FastHistogramCZ(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    np.ndarray[dtype=np.int32_t, ndim=1,
                               negative_indices=False, mode='c'] hist,
                    float radius):
    cdef:
        int w, h, w2, h2, y, x

    h = im.shape[0]
    w = im.shape[1]
    w2 = w / 2
    h2 = h / 2
    for y in range(h):
        for x in range(w):
            if sqrt(((y - h2) * (y - h2)) + ((x - w2) * (x - w2))) < radius:
                if im[y, x] < 0:
                    hist[0] += 1
                else:
                    hist[<int> im[y, x]] += 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FastHistogramDisc(np.ndarray[dtype=np.float32_t, ndim=2,
                                 negative_indices=False, mode='c'] im,
                      np.ndarray[dtype=np.int32_t, ndim=1,
                                 negative_indices=False, mode='c'] hist,
                      int h2, int w2, float radius):
    cdef:
        int w, h, y, x

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            if sqrt(((y - h2) * (y - h2)) + ((x - w2) * (x - w2))) < radius:
                if im[y, x] < 0:
                    hist[0] += 1
                else:
                    hist[<int> im[y, x]] += 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
def SmoothCZ(np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] im,
             np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] smoothed,
             int start, int stop,
             np.ndarray[dtype=np.int32_t, ndim=3,
                        negative_indices=False, mode='c'] coords,
             np.ndarray[dtype=np.int32_t, ndim=1,
                        negative_indices=False, mode='c'] coord_count,
             np.ndarray[dtype=np.float32_t, ndim=1,
                        negative_indices=False, mode='c'] scratch,
             np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] dist,
             float corner_r,
             np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] middle_smooth,
             ):
    cdef:
        int w, h, y, x, r, p, i, j, smooth_len, s2, i2, i21, fw
        float mean_total, mean_count,

    h = im.shape[0]
    w = im.shape[1]
    smooth_len = scratch.shape[0]
    assert smooth_len % 2 == 1
    s2 = (smooth_len - 1) / 2
    fw = <int> sqrt(smooth_len - 1)

    # smooth rounded corners
    for r in range(stop - start):
        mean_total = 0
        for i in range(smooth_len):
            scratch[i] = im[coords[r, i, 0], coords[r, i, 1]]
            mean_total += scratch[i]

        for i in range(smooth_len, coord_count[r] + smooth_len):
            p = (i - s2 - 1) % coord_count[r]
            i2 = i % coord_count[r]
            i21 = (i2 - 1) % coord_count[r]
            if smoothed[coords[r, p, 0], coords[r, p, 1]] == 0:
                smoothed[coords[r, p, 0], coords[r, p, 1]] = mean_total / float(smooth_len)
            mean_total -= scratch[i % smooth_len]
            scratch[i % smooth_len] = im[coords[r, i2, 0], coords[r, i2, 1]]
            mean_total += im[coords[r, i2, 0], coords[r, i2, 1]]

            if (coords[r, i2, 0] != coords[r, i21, 0] and
                        coords[r, i2, 1] != coords[r, i21, 1]):
                pass

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if smoothed[y, x] == 0.0:
                if (dist[y, x] > corner_r and
                            smoothed[y - 1, x] > 0 and smoothed[y + 1, x] > 0 and
                            smoothed[y, x - 1] > 0 and smoothed[y, x + 1] > 0):
                    # fill corner holes
                    smoothed[y, x] = (smoothed[y - 1, x] +
                                      smoothed[y, x - 1] +
                                      smoothed[y, x + 1] +
                                      smoothed[y + 1, x]) / 4.0
                else:
                    # wafer middle
                    smoothed[y, x] = middle_smooth[y, x]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def GetCCCoords(np.ndarray[dtype=np.int32_t, ndim=2,
                           negative_indices=False, mode='c'] ccs,
                np.ndarray[dtype=np.int32_t, ndim=3,
                           negative_indices=False, mode='c'] coords,
                np.ndarray[dtype=np.int32_t, ndim=1,
                           negative_indices=False, mode='c'] counts,
                ):
    cdef:
        int w, h, x, y

    h, w = ccs.shape[0], ccs.shape[1]

    for y in range(h):
        for x in range(w):
            if ccs[y, x] > 0:
                coords[ccs[y, x], counts[ccs[y, x]], 0] = y
                coords[ccs[y, x], counts[ccs[y, x]], 1] = x
                counts[ccs[y, x]] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def EdgeCandidates(np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] im,
                   np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] corners,
                   np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] filtered,
                   np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] edges):
    cdef:
        int w, h, y, x, a, b, num_points, i, mid_x, mid_y, dy, dx, w3, h3
        int steep, sx, sy, d, j, dist, y1, y2, x1, x2
        int counts[4], count, min_d, max_d
        float t
        int corner_index = -1

    h, w = im.shape[0], im.shape[1]

    # create filtered edge image (pick filter based on corner location)
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if corners[y, x] == 1:
                filtered[y, x] = im[y - 1, x - 1] - im[y + 1, x + 1]
            elif corners[y, x] == 2:
                filtered[y, x] = im[y - 1, x + 1] - im[y + 1, x - 1]
            elif corners[y, x] == 3:
                filtered[y, x] = im[y + 1, x - 1] - im[y - 1, x + 1]
            elif corners[y, x] == 4:
                filtered[y, x] = im[y + 1, x + 1] - im[y - 1, x - 1]

    # find the strongest edge gradients
    t = 0.025
    for y in range(3, h - 3):
        for x in range(3, w - 3):
            if filtered[y, x] > t:
                if (corners[y, x] == 1) and filtered[y, x] > filtered[y - 1, x - 1] and filtered[y, x] > filtered[
                            y + 1, x + 1]:
                    edges[y, x] = 1
                elif (corners[y, x] == 2) and filtered[y, x] > filtered[y + 1, x - 1] and filtered[y, x] > filtered[
                            y - 1, x + 1]:
                    edges[y, x] = 1
                elif (corners[y, x] == 3) and filtered[y, x] > filtered[y + 1, x - 1] and filtered[y, x] > filtered[
                            y - 1, x + 1]:
                    edges[y, x] = 1
                elif (corners[y, x] == 4) and filtered[y, x] > filtered[y - 1, x - 1] and filtered[y, x] > filtered[
                            y + 1, x + 1]:
                    edges[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def AccumulateCircleCenter(np.ndarray[dtype=np.int32_t, ndim=3,
                                      negative_indices=False, mode='c'] coords,
                           np.ndarray[dtype=np.int32_t, ndim=1,
                                      negative_indices=False, mode='c'] counts,
                           np.ndarray[dtype=np.int32_t, ndim=2,
                                      negative_indices=False, mode='c'] acc,
                           int h, int w):
    cdef:
        int acc_wh, min_d, max_d, num_cc, i, num_points, a, b, h2, count
        int steep, sx, sy, d, j, dist, y1, y2, x1, x2, y, x, dy, dx

    acc_wh = min(acc.shape[0], acc.shape[1])
    num_cc = coords.shape[0]

    min_d = 100
    max_d = h + w
    h2 = h // 2

    # draw lines for potential centers
    for i in range(1, num_cc):
        num_points = counts[i]
        for a in range(num_points):
            for b in range(a + 1, num_points):
                y1, x1 = coords[i, a, 0], coords[i, a, 1]
                y2, x2 = coords[i, b, 0], coords[i, b, 1]
                y = ((y1 + y2) // 2) + h
                x = ((x1 + x2) // 2) + w
                dy = y2 - y1
                dx = x2 - x1

                # ignore horizontal and vertical lines
                if dy == 0 or dx == 0: continue

                # make sure at least 5-100 pixels apart
                dist = (dy ** 2) + (dx ** 2)
                if dist < 25: continue

                # draw line from (y, x) to acc edge in direction perpendicular
                #  to original points vector
                dy, dx = dx, -dy  # perpendicular

                # make sure dy pointing down if in top, or up if in bottom
                if ((y1 < h2 and dy < 0) or (y1 > h2 and dy > 0)):
                    dy *= -1
                    dx *= -1

                if dx > 0:
                    sx = 1
                else:
                    sx = -1
                if dy > 0:
                    sy = 1
                else:
                    sy = -1

                dy, dx = abs(dy), abs(dx)
                if dy > dx:
                    steep = 1
                    y, x = x, y
                    dy, dx = dx, dy
                    sx, sy = sy, sx
                else:
                    steep = 0

                d = (2 * dy) - dx
                count = 0
                while (x > 0) and (x < acc_wh) and (y > 0) and (y < acc_wh) and count < max_d:
                    if count > min_d:
                        if steep == 1:
                            acc[x, y] += 1
                        else:
                            acc[y, x] += 1

                    while d >= 0:
                        y += sy
                        d -= (2 * dx)
                    x += sx
                    d += 2 * dy
                    count += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def EdgeCoordsCZ(np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] dist,
                 np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] theta,
                 int start, int stop,
                 np.ndarray[dtype=np.float32_t, ndim=1,
                            negative_indices=False, mode='c'] corners,
                 np.ndarray[dtype=np.int32_t, ndim=3,
                            negative_indices=False, mode='c'] coords,
                 np.ndarray[dtype=np.int32_t, ndim=1,
                            negative_indices=False, mode='c'] coord_count):
    cdef:
        int w, h, y, x, r, r_count  #, mean_count
        float d, diff1, diff2, diff3, corners2[4]

    h = dist.shape[0]
    w = dist.shape[1]

    # rounded corners first
    for r in range(stop - start):
        y = r + start
        x = 0
        r_count = 0

        # top
        while theta[y, x] < corners[7]:
            x += 1
        while theta[y, x] < corners[0]:
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
            x += 1
        x -= 1

        # top right
        d = dist[y, x]
        while theta[y, x] < corners[1]:
            diff1 = abs(dist[y, x + 1] - d)
            diff2 = abs(dist[y + 1, x] - d)
            diff3 = abs(dist[y + 1, x + 1] - d)
            if diff1 < diff2 and diff1 < diff3:
                x += 1
            elif diff2 < diff1 and diff2 < diff3:
                y += 1
            else:
                y += 1
                x += 1
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
        x -= 1

        # right
        while theta[y, x] < corners[2]:
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
            y += 1
        y -= 1

        # bottom right
        d = dist[y, x]
        while theta[y, x] < corners[3]:
            diff1 = abs(dist[y, x - 1] - d)
            diff2 = abs(dist[y + 1, x] - d)
            diff3 = abs(dist[y + 1, x - 1] - d)
            if diff1 < diff2 and diff1 < diff3:
                x -= 1
            elif diff2 < diff1 and diff2 < diff3:
                y += 1
            else:
                y += 1
                x -= 1
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
        y -= 1

        # bottom
        while theta[y, x] < corners[4]:
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
            x -= 1
        x += 1

        # bottom left
        d = dist[y, x]
        while theta[y, x] < corners[5]:
            diff1 = abs(dist[y, x - 1] - d)
            diff2 = abs(dist[y - 1, x] - d)
            diff3 = abs(dist[y - 1, x - 1] - d)
            if diff1 < diff2 and diff1 < diff3:
                x -= 1
            elif diff2 < diff1 and diff2 < diff3:
                y -= 1
            else:
                y -= 1
                x -= 1
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
        x += 1

        # left
        while theta[y, x] > 0 or theta[y, x] < corners[6]:
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1
            y -= 1
        y += 1

        # top left
        d = dist[y, x]
        while theta[y, x] < corners[7]:
            diff1 = abs(dist[y, x + 1] - d)
            diff2 = abs(dist[y - 1, x] - d)
            diff3 = abs(dist[y - 1, x + 1] - d)
            if diff1 < diff2 and diff1 < diff3:
                x += 1
            elif diff2 < diff1 and diff2 < diff3:
                y -= 1
            else:
                y -= 1
                x += 1
            coords[r, r_count, 0] = y
            coords[r, r_count, 1] = x
            r_count += 1

        coord_count[r] = r_count

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def RDS(np.ndarray[dtype=np.float32_t, ndim=2,
                   negative_indices=False, mode='c'] im,
        float radius,
        float inner_edge,
        float outer_edge):
    cdef:
        int w, h, w2, h2, y, x
        float d, outer_total, inner_total
        float outer_count, inner_count

    outer_total = 0
    inner_total = 0
    outer_count = 0
    inner_count = 0

    h = im.shape[0]
    w = im.shape[1]
    w2 = w / 2
    h2 = h / 2
    for y in range(h):
        for x in range(w):
            d = sqrt(((y - h2) * (y - h2)) + ((x - w2) * (x - w2)))
            if d <= radius:
                if d > outer_edge:
                    outer_total += im[y, x]
                    outer_count += 1
                elif d <= inner_edge:
                    inner_total += im[y, x]
                    inner_count += 1

    if inner_count > 0 and outer_count > 0:
        return (inner_total / inner_count), (outer_total / outer_count)
    else:
        return 1, 1

@cython.boundscheck(False)
@cython.wraparound(False)
def RDSlug(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im,
           float radius, float inner_edge, int h2, int w2):
    cdef:
        int w, h, y, x
        float d, outer_total, inner_total
        float outer_count, inner_count

    outer_total = 0
    inner_total = 0
    outer_count = 0
    inner_count = 0

    h = im.shape[0]
    w = im.shape[1]
    for y in range(h):
        for x in range(w):
            d = sqrt(((y - h2) * (y - h2)) + ((x - w2) * (x - w2)))
            if d <= radius:
                if d > inner_edge:
                    outer_total += im[y, x]
                    outer_count += 1
                else:
                    inner_total += im[y, x]
                    inner_count += 1

    if inner_count > 0 and outer_count > 0:
        return (inner_total / inner_count), (outer_total / outer_count)
    else:
        return 1, 1

@cython.boundscheck(False)
@cython.wraparound(False)
def GradientMaximums(np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] im,
                     np.ndarray[dtype=np.uint8_t, ndim=2,
                                negative_indices=False, mode='c'] maximums,
                     float edge_thresh, float ratio_thresh
                     ):
    cdef:
        int w, h, x, y, i, j, r, max_index
        float local_or[8], max_val, v1, v2, r1, r2

    h = im.shape[0]
    w = im.shape[1]

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if im[y, x] < edge_thresh: continue

            # find local orentation
            local_or[0] = im[y - 1, x - 1] + im[y + 1, x + 1]
            local_or[1] = im[y - 1, x] + im[y + 1, x]
            local_or[2] = im[y - 1, x + 1] + im[y + 1, x - 1]
            local_or[3] = im[y, x + 1] + im[y, x - 1]
            local_or[4] = im[y + 1, x + 1] + im[y - 1, x - 1]
            local_or[5] = im[y + 1, x] + im[y - 1, x]
            local_or[6] = im[y + 1, x - 1] + im[y - 1, x + 1]
            local_or[7] = im[y, x - 1] + im[y, x + 1]
            max_val = 0
            for r in range(8):
                if local_or[r] > max_val:
                    max_val = local_or[r]
                    max_index = r

            if max_index == 2:
                r1 = im[y - 1, x + 1]
                r2 = im[y + 1, x - 1]
                v1 = im[y + 1, x + 1]
                v2 = im[y - 1, x - 1]
            elif max_index == 3:
                r1 = im[y, x + 1]
                r2 = im[y, x - 1]
                v1 = im[y + 1, x]
                v2 = im[y - 1, x]
            elif max_index == 4:
                r1 = im[y + 1, x + 1]
                r2 = im[y - 1, x - 1]
                v1 = im[y + 1, x - 1]
                v2 = im[y - 1, x + 1]
            elif max_index == 5:
                r1 = im[y + 1, x]
                r2 = im[y - 1, x]
                v1 = im[y, x - 1]
                v2 = im[y, x + 1]
            elif max_index == 6:
                r1 = im[y + 1, x - 1]
                r2 = im[y - 1, x + 1]
                v1 = im[y - 1, x - 1]
                v2 = im[y + 1, x + 1]
            elif max_index == 7:
                r1 = im[y, x - 1]
                r2 = im[y, x + 1]
                v1 = im[y - 1, x]
                v2 = im[y + 1, x]
            elif max_index == 0:
                r1 = im[y - 1, x - 1]
                r2 = im[y + 1, x + 1]
                v1 = im[y - 1, x + 1]
                v2 = im[y + 1, x - 1]
            elif max_index == 1:
                r1 = im[y - 1, x]
                r2 = im[y + 1, x]
                v1 = im[y, x + 1]
                v2 = im[y, x - 1]

            if v1 > im[y, x] or v2 > im[y, x]: continue
            if r1 < 0.001 or r2 < 0.001: continue
            if (max(r1, r2) / min(r1, r2)) - 1 > ratio_thresh: continue

            maximums[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def SlipperDistances(np.ndarray[dtype=np.uint8_t, ndim=2,
                                negative_indices=False, mode='c'] slippers,
                     np.ndarray[dtype=np.int32_t, ndim=2,
                                negative_indices=False, mode='c'] cc_labels,
                     np.ndarray[dtype=np.float32_t, ndim=1,
                                negative_indices=False, mode='c'] max_ds):
    cdef:
        int w, h, x, y, xc, yc
        float d


    h = slippers.shape[0]
    w = slippers.shape[1]
    xc = w // 2
    yc = h // 2

    for y in range(h):
        for x in range(w):
            if cc_labels[y, x] == 0: continue
            d = sqrt(((x - xc) * (x - xc)) + ((y - yc) * (y - yc)))
            if d > max_ds[cc_labels[y, x]]:
                max_ds[cc_labels[y, x]] = d

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def SlipEnhance(np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] im,
                np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] filtered,
                int length, int width,
                np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] scratch, ):
    cdef:
        int w, h, x, y, i, j  #, xc, yc, d2
        float means[6]


    h = im.shape[0]
    w = im.shape[1]

    for y in range(length + width, h - length - width):
        for x in range(length + width, w - length - width):
            for i in range(-length, length + 1):
                # \
                scratch[0, i + length] = im[y + i + width, x + i - width]
                scratch[1, i + length] = im[y + i, x + i]
                scratch[2, i + length] = im[y + i - width, x + i + width]

                scratch[3, i + length] = im[y - i - width, x + i - width]
                scratch[4, i + length] = im[y - i, x + i]
                scratch[5, i + length] = im[y - i + width, x + i + width]

            for i in range(6):
                means[i] = 0
                for j in range(length):
                    means[i] += scratch[i, j]
                means[i] /= length

            filtered[y, x] = max(min(means[0], means[2]) - means[1],
                                 min(means[3], means[5]) - means[4],
                                 0)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CrackEnhance(np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] im,
                 np.ndarray[dtype=np.float32_t, ndim=2,
                            negative_indices=False, mode='c'] straight,
                 #np.ndarray[dtype=np.float32_t, ndim=2,
                 #           negative_indices=False, mode='c'] cross,
                 ):
    cdef:
        int w, h, x, y, r
        int min_index = 0
        float max_val, min_val, v
        float local_vals[16]
        float op_vals[8]


    h = im.shape[0]
    w = im.shape[1]

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            # find local orentation
            local_vals[0] = im[y - 2, x - 2]
            local_vals[1] = im[y - 2, x - 1]
            local_vals[2] = im[y - 2, x]
            local_vals[3] = im[y - 2, x + 1]
            local_vals[4] = im[y - 2, x + 2]
            local_vals[5] = im[y - 1, x + 2]
            local_vals[6] = im[y, x + 2]
            local_vals[7] = im[y + 1, x + 2]
            local_vals[8] = im[y + 2, x + 2]
            local_vals[9] = im[y + 2, x + 1]
            local_vals[10] = im[y + 2, x]
            local_vals[11] = im[y + 2, x - 1]
            local_vals[12] = im[y + 2, x - 2]
            local_vals[13] = im[y + 1, x - 2]
            local_vals[14] = im[y, x - 2]
            local_vals[15] = im[y - 1, x - 2]

            min_val = 1000
            max_val = 0
            for r in range(8):
                op_vals[r] = local_vals[r] + local_vals[r + 8]
                if op_vals[r] < min_val:
                    min_val = op_vals[r]
                    min_index = r
                    #if op_vals[r] > max_val: max_val = op_vals[r]

            v = (min(local_vals[min_index + 4], local_vals[(min_index + 12) % 16]) - im[y, x])
            if v > 0: straight[y, x] = v

            #v = max_val - ((local_vals[min_index] + local_vals[min_index+4] +
            #                local_vals[min_index+8] + local_vals[(min_index+12)%16]) / 4.0)
            #if v > 0: cross[y, x] = v

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CrackEnhance2(np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] im,
                  np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] straight,
                  ):
    cdef:
        int w, h, x, y, r
        int min_index = 0
        float max_val, min_val, v
        float local_vals[16]
        float op_vals[8]


    h = im.shape[0]
    w = im.shape[1]

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            # find local orentation
            local_vals[0] = im[y - 2, x - 2]
            local_vals[1] = im[y - 2, x - 1]
            local_vals[2] = im[y - 2, x]
            local_vals[3] = im[y - 2, x + 1]
            local_vals[4] = im[y - 2, x + 2]
            local_vals[5] = im[y - 1, x + 2]
            local_vals[6] = im[y, x + 2]
            local_vals[7] = im[y + 1, x + 2]
            local_vals[8] = im[y + 2, x + 2]
            local_vals[9] = im[y + 2, x + 1]
            local_vals[10] = im[y + 2, x]
            local_vals[11] = im[y + 2, x - 1]
            local_vals[12] = im[y + 2, x - 2]
            local_vals[13] = im[y + 1, x - 2]
            local_vals[14] = im[y, x - 2]
            local_vals[15] = im[y - 1, x - 2]

            min_val = 1000
            max_val = 0
            for r in range(8):
                op_vals[r] = local_vals[r] + local_vals[r + 8]
                if op_vals[r] < min_val:
                    min_val = op_vals[r]
                    min_index = r
                    #if op_vals[r] > max_val: max_val = op_vals[r]

            v = (min(local_vals[min_index + 4], local_vals[(min_index + 12) % 16]) / max(0.01, im[y, x]))
            if v > 1.0: straight[y, x] = v - 1.0

            #v = max_val - ((local_vals[min_index] + local_vals[min_index+4] +
            #                local_vals[min_index+8] + local_vals[(min_index+12)%16]) / 4.0)
            #if v > 0: cross[y, x] = v

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def InvertedRidgeEnhance(np.ndarray[dtype=np.float32_t, ndim=2,
                                    negative_indices=False, mode='c'] im,
                         np.ndarray[dtype=np.float32_t, ndim=2,
                                    negative_indices=False, mode='c'] filtered,
                         ):
    cdef:
        int w, h, x, y, r
        int max_index = 0
        float max_val, v
        float local_vals[16]


    h = im.shape[0]
    w = im.shape[1]

    for y in range(2, h - 2):
        for x in range(2, w - 2):
            # find local orentation
            local_vals[0] = im[y - 2, x - 2]
            local_vals[1] = im[y - 2, x - 1]
            local_vals[2] = im[y - 2, x]
            local_vals[3] = im[y - 2, x + 1]
            local_vals[4] = im[y - 2, x + 2]
            local_vals[5] = im[y - 1, x + 2]
            local_vals[6] = im[y, x + 2]
            local_vals[7] = im[y + 1, x + 2]
            local_vals[8] = im[y + 2, x + 2]
            local_vals[9] = im[y + 2, x + 1]
            local_vals[10] = im[y + 2, x]
            local_vals[11] = im[y + 2, x - 1]
            local_vals[12] = im[y + 2, x - 2]
            local_vals[13] = im[y + 1, x - 2]
            local_vals[14] = im[y, x - 2]
            local_vals[15] = im[y - 1, x - 2]

            max_val = -1000
            for r in range(8):
                if local_vals[r] + local_vals[r + 8] > max_val:
                    max_val = local_vals[r] + local_vals[r + 8]
                    max_index = r

            if False and x == 21 and y == 393:
                print max_index
                print im[y, x], local_vals[max_index], local_vals[max_index + 8]
                print max_val
                return

            # find difference between ridge intensity and local background
            v = (min(im[y, x], local_vals[max_index], local_vals[max_index + 8]) -
                 max(local_vals[max_index + 4], local_vals[(max_index + 12) % 16]))

            #v = (((im[y, x] + local_vals[max_index] + local_vals[max_index+8]) / 3.0) -
            #     ((local_vals[max_index+4] + local_vals[(max_index+12)%16]) / 2.0))

            if v > 0: filtered[y, x] = v
            #filtered[y, x] = v

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CCSizes(np.ndarray[dtype=np.int32_t, ndim=2,
                       negative_indices=False, mode='c'] cc_labels,
            np.ndarray[dtype=np.int32_t, ndim=1,
                       negative_indices=False, mode='c'] cc_sizes):
    cdef:
        unsigned int w, h, x, y

    h = <unsigned int> cc_labels.shape[0]
    w = <unsigned int> cc_labels.shape[1]

    for y in range(h):
        for x in range(w):
            if cc_labels[y, x] != 0:
                cc_sizes[cc_labels[y, x]] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def GrainProps(np.ndarray[dtype=np.int32_t, ndim=2,
                          negative_indices=False, mode='c'] cc_labels,
               np.ndarray[dtype=np.int32_t, ndim=1,
                          negative_indices=False, mode='c'] cc_sizes,
               np.ndarray[dtype=np.float64_t, ndim=2,
                          negative_indices=False, mode='c'] im,
               np.ndarray[dtype=np.float64_t, ndim=1,
                          negative_indices=False, mode='c'] totals):
    cdef:
        unsigned int w, h, x, y

    h = <unsigned int> cc_labels.shape[0]
    w = <unsigned int> cc_labels.shape[1]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if cc_labels[y, x] != 0:
                cc_sizes[cc_labels[y, x]] += 1

                totals[cc_labels[y, x]] += im[y, x]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FilterRegions(np.ndarray[dtype=np.int32_t, ndim=2,
                             negative_indices=False, mode='c'] cc_labels,
                  np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] mask,
                  np.ndarray[dtype=np.int32_t, ndim=1,
                             negative_indices=False, mode='c'] cc_sizes,
                  np.ndarray[dtype=np.float64_t, ndim=1,
                             negative_indices=False, mode='c'] cc_total_orig,
                  np.ndarray[dtype=np.float64_t, ndim=1,
                             negative_indices=False, mode='c'] cc_total_smooth,
                  int lower_size_thresh,
                  int upper_size_thresh,
                  float intensity_diff_thresh):
    cdef:
        unsigned int w, h, x, y

    h = <unsigned int> cc_labels.shape[0]
    w = <unsigned int> cc_labels.shape[1]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0: continue
            if cc_sizes[cc_labels[y, x]] < lower_size_thresh:
                mask[y, x] = 0
                continue
            if cc_sizes[cc_labels[y, x]] > upper_size_thresh: continue
            if (cc_total_orig[cc_labels[y, x]] > 0.4 and
                        cc_total_orig[cc_labels[y, x]] > cc_total_smooth[
                        cc_labels[y, x]] - intensity_diff_thresh): continue

            mask[y, x] = 0

    return True

@cython.boundscheck(False)
@cython.wraparound(False)
def RegionProps(np.ndarray[dtype=np.int32_t, ndim=2,
                           negative_indices=False, mode='c'] cc_labels,
                np.ndarray[dtype=np.int32_t, ndim=1,
                           negative_indices=False, mode='c'] cc_sizes,
                np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] im,
                np.ndarray[dtype=np.float64_t, ndim=1,
                           negative_indices=False, mode='c'] totals):
    cdef:
        unsigned int w, h, x, y

    h = <unsigned int> cc_labels.shape[0]
    w = <unsigned int> cc_labels.shape[1]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if cc_labels[y, x] != 0:
                cc_sizes[cc_labels[y, x]] += 1

                totals[cc_labels[y, x]] += im[y, x]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def RemoveSmallCCs(np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] candidates,
                   np.ndarray[dtype=np.int32_t, ndim=2,
                              negative_indices=False, mode='c'] cc_labels,
                   np.ndarray[dtype=np.int32_t, ndim=1,
                              negative_indices=False, mode='c'] cc_sizes,
                   int min_size):
    cdef:
        unsigned int w, h, x, y

    h = <unsigned int> cc_labels.shape[0]
    w = <unsigned int> cc_labels.shape[1]

    for y in range(h):
        for x in range(w):
            if cc_sizes[cc_labels[y, x]] < min_size:
                candidates[y, x] = 0

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def RemoveRangeCC(np.ndarray[dtype=np.int32_t, ndim=2,
                             negative_indices=False, mode='c'] candidates,
                  np.ndarray[dtype=np.int32_t, ndim=2,
                             negative_indices=False, mode='c'] cc_labels,
                  np.ndarray[dtype=np.int32_t, ndim=1,
                             negative_indices=False, mode='c'] cc_sizes,
                  int min_size, int max_size):
    cdef:
        unsigned int w, h, x, y

    h = <unsigned int> cc_labels.shape[0]
    w = <unsigned int> cc_labels.shape[1]

    for y in range(h):
        for x in range(w):
            if cc_sizes[cc_labels[y, x]] < min_size or cc_sizes[cc_labels[y, x]] > max_size:
                candidates[y, x] = 0

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ImpureRegionAnalysis(np.ndarray[dtype=np.uint8_t, ndim=2,
                                    negative_indices=False, mode='c'] mask,
                         np.ndarray[dtype=np.float32_t, ndim=1,
                                    negative_indices=False, mode='c'] region_counts,
                         int edge_width,
                         int corner_width
                         ):
    cdef:
        int x, y, w, h,
        float avg_dist

    h = mask.shape[0]
    w = mask.shape[1]

    # ignore pixels along the edges because these is a high chance of false
    #  positives
    avg_dist = 0
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x] == 0: continue

            avg_dist += min(x, y, w - x - 1, h - y - 1)

            # top edge
            if y < edge_width: region_counts[0] += 1

            # left edge
            if x < edge_width: region_counts[1] += 1

            # right edge
            if x >= (w - edge_width): region_counts[2] += 1

            # bottom edge
            if y >= (h - edge_width): region_counts[3] += 1

            # top right corner
            if ((y < corner_width) or (x >= (w - corner_width))):
                region_counts[4] += 1

            # top left corner
            if ((y < corner_width) or (x < corner_width)):
                region_counts[5] += 1

            # bottom right corner
            if ((y >= (h - corner_width)) or (x >= (w - corner_width))):
                region_counts[6] += 1

            # bottom left corner
            if ((y >= (h - corner_width)) or (x < corner_width)):
                region_counts[7] += 1

    return avg_dist

@cython.boundscheck(False)
@cython.wraparound(False)
def CircleHoughAcc2(np.ndarray[dtype=np.int32_t, ndim=1,
                               negative_indices=False, mode='c'] ys,
                    np.ndarray[dtype=np.int32_t, ndim=1,
                               negative_indices=False, mode='c'] xs,
                    np.ndarray[dtype=np.int32_t, ndim=1,
                               negative_indices=False, mode='c'] acc_ys,
                    np.ndarray[dtype=np.int32_t, ndim=1,
                               negative_indices=False, mode='c'] acc_xs,
                    np.ndarray[dtype=np.int32_t, ndim=3,
                               negative_indices=False, mode='c'] acc,
                    int min_r, int max_r):
    cdef:
        int x, y, w, h, d, p, i, j, num_points, num_y, num_x, r

    num_x = acc_xs.shape[0]
    num_y = acc_ys.shape[0]
    num_points = ys.shape[0]
    for p in range(num_points):
        for i in range(num_y):
            for j in range(num_x):
                r = <int> round(sqrt(((ys[p] - acc_ys[i]) ** 2 + (xs[p] - acc_xs[j]) ** 2)))
                if min_r < r < max_r:
                    acc[i, j, r - min_r] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CircleHoughAcc(np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] rings,
                   np.ndarray[dtype=np.int32_t, ndim=1,
                              negative_indices=False, mode='c'] acc):
    cdef:
        int x, y, w, h, d
        int middle_x, middle_y

    h = rings.shape[0]
    w = rings.shape[1]
    middle_y = h // 2
    middle_x = w // 2

    for y in range(h):
        for x in range(w):
            if rings[y, x] == 0: continue

            # compute distance
            d = <int> round(sqrt(((y - middle_y) * (y - middle_y)) +
                                 ((x - middle_x) * (x - middle_x))))
            acc[d] += 1
    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CombineBreaks(np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] breaks,
                  np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] break_counts,
                  np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] edge_strength):
    cdef:
        int x, y, w, h, d, mode, start, stop, max_pos
        float max_val

    h = breaks.shape[0]
    w = breaks.shape[1]
    for y in range(h):
        x = 0
        while x < w:
            if break_counts[y, x] > 0:
                max_val = -1
                max_pos = x
                while x < w and break_counts[y, x] > 0:
                    breaks[y, x] = 0
                    if edge_strength[y, x] > max_val:
                        max_val = edge_strength[y, x]
                        max_pos = x
                    x += 1
                breaks[y, max_pos] = 1
            else:
                x += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def LineMidPoints(np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] im_in,
                  np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] im_out):
    cdef:
        int x, y, w, h, d, mode, start, stop

    h = im_in.shape[0]
    w = im_in.shape[1]
    for y in range(h):
        mode = 0
        for x in range(w):
            if mode == 0 and im_in[y, x] == 1:
                start = x
                mode = 1
            elif mode == 1 and im_in[y, x] == 0:
                stop = x
                mode = 0
                im_out[y, (start + stop) // 2] = 1

@cython.boundscheck(False)
@cython.wraparound(False)
def HVProfile(np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] im,
              np.ndarray[dtype=np.float32_t, ndim=1,
                         negative_indices=False, mode='c'] v_profile,
              np.ndarray[dtype=np.float32_t, ndim=1,
                         negative_indices=False, mode='c'] h_profile):
    cdef:
        int x, y, w, h, s, v_count, h_count, d
        float count, total

    h = im.shape[0]
    w = im.shape[1]
    s = (v_profile.shape[0] - 1) // 2
    v_count, h_count = 0, 0

    for y in range(s, h - s):
        for x in range(s, w - s):
            #if (im[y, x] < im[y-1, x] and im[y, x] < im[y+1, x]):
            if (im[y, x] < im[y - 1, x] and im[y, x] < im[y + 1, x] and
                        im[y, x] < im[y, x - 1] and im[y, x] < im[y, x + 1] and
                        im[y, x] < im[y - 1, x - 1] and im[y, x] < im[y + 1, x - 1] and
                        im[y, x] < im[y + 1, x + 1] and im[y, x] < im[y - 1, x + 1]):
                v_count += 1
                for d in range(-s, s + 1):
                    v_profile[s + d] += im[y + d, x] - im[y, x]

                    #if (im[y, x] < im[y, x-1] and im[y, x] < im[y, x+1]):
                h_count += 1
                for d in range(-s, s + 1):
                    h_profile[s + d] += im[y, x + d] - im[y, x]

    v_profile /= v_count
    h_profile /= h_count

@cython.boundscheck(False)
@cython.wraparound(False)
def RowAvgs(np.ndarray[dtype=np.float32_t, ndim=2,
                       negative_indices=False, mode='c'] im,
            np.ndarray[dtype=np.uint8_t, ndim=2,
                       negative_indices=False, mode='c'] mask,
            np.ndarray[dtype=np.float32_t, ndim=1,
                       negative_indices=False, mode='c'] row_avgs,
            int min_count):
    cdef:
        int x, y, w, h
        float count, total

    h = mask.shape[0]
    w = mask.shape[1]

    for y in range(h):
        count = 0
        total = 0
        for x in range(w):
            if mask[y, x] == 1:
                total += im[y, x]
                count += 1
        if count > min_count:
            row_avgs[y] = total / count

@cython.boundscheck(False)
@cython.wraparound(False)
def FindCellMiddle(np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] mask):
    cdef:
        int x, y, w, h, in_mask, s1, s2

    h = mask.shape[0]
    w = mask.shape[1]

    for y in range(h):
        in_mask = 1
        for x in range(1, w - 1):
            if in_mask == 1 and mask[y, x] == 1:
                continue
            elif in_mask == 1 and mask[y, x] == 0:
                in_mask = 0
                s1 = x
            elif in_mask == 0 and mask[y, x] == 1:
                s2 = x
                mask[y, (s1 + s2) // 2] = 2
                in_mask = 1

@cython.boundscheck(False)
@cython.wraparound(False)
def SumBrightLines(np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] edges_dampened,
                   np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] bright_lines,
                   np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] break_stength,
                   float edge_thresh, int sum_length
                   ):
    cdef:
        int x, y, w, h, s, x2, count
        float total

    h = edges_dampened.shape[0]
    w = edges_dampened.shape[1]

    for y in range(h):
        for x in range(1, w - 1):
            if edges_dampened[y, x] < edge_thresh: continue
            total = 0
            count = 0
            if bright_lines[y, x - 1] > bright_lines[y, x + 1]:
                for x2 in range(x, max(0, x - sum_length), -1):
                    total += bright_lines[y, x2]
                    count += 1
            else:
                for x2 in range(x, min(w, x + sum_length)):
                    total += bright_lines[y, x2]
                    count += 1

            if count > 10: break_stength[y, x] = total / count

@cython.boundscheck(False)
@cython.wraparound(False)
def DampenEdges(np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] edges_in,
                np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] edges_out,
                int wm
                ):
    cdef:
        int x, y, w, h, s, i  #, s_top_right, s_top_left, s_bottom_right, s_bottom_left
        float max_top_left, max_top_right, max_bottom_left, max_bottom_right

    h = edges_in.shape[0]
    w = edges_in.shape[1]

    for y in range(1, h - 1):
        for x in range(w):
            if edges_in[y, x - 1] < edges_in[y, x] > edges_in[y, x + 1]:
                s = min(wm, x, w - 1 - x)
                max_top_left, max_top_right, max_bottom_left, max_bottom_right = 0, 0, 0, 0
                for i in range(s + 1):
                    if edges_in[y - 1, x + i] > max_top_right:
                        max_top_right = edges_in[y - 1, x + i]
                    if edges_in[y - 1, x - i] > max_top_left:
                        max_top_left = edges_in[y - 1, x - i]
                    if edges_in[y + 1, x + i] > max_bottom_right:
                        max_bottom_right = edges_in[y + 1, x + i]
                    if edges_in[y + 1, x - i] > max_bottom_left:
                        max_bottom_left = edges_in[y + 1, x - i]

                edges_out[y, x] = edges_in[y, x] - max(min(max_top_left, max_bottom_right),
                                                       min(max_top_right, max_bottom_left))

@cython.boundscheck(False)
@cython.wraparound(False)
def RadialNormalise(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    np.ndarray[dtype=np.float32_t, ndim=1,
                               negative_indices=False, mode='c'] profile,
                    int y_center, int x_center
                    ):
    cdef:
        int x, y, w, h, r, y2

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        y2 = (y - y_center) ** 2
        for x in range(w):
            r = <int> round(sqrt(y2 + (x - x_center) ** 2))
            im[y, x] -= profile[r]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def RadialFill(np.ndarray[dtype=np.float32_t, ndim=2,
                          negative_indices=False, mode='c'] im,
               np.ndarray[dtype=np.float32_t, ndim=1,
                          negative_indices=False, mode='c'] profile,
               int r_thresh, int edge_width, int y_center, int x_center):
    cdef:
        int x, y, w, h, r, y2

    h = im.shape[0]
    w = im.shape[1]

    for y in range(edge_width, h - edge_width, 1):
        y2 = (y - y_center) ** 2
        for x in range(edge_width, w - edge_width, 1):
            r = <int> round(sqrt(y2 + (x - x_center) ** 2))
            if r < r_thresh: continue
            im[y, x] = profile[r]

@cython.boundscheck(False)
@cython.wraparound(False)
def RadialMaxes(np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] im,
                np.ndarray[dtype=np.float32_t, ndim=1,
                           negative_indices=False, mode='c'] maxes,
                int y_center, int x_center
                ):
    cdef:
        int x, y, w, h, r, y2

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        y2 = (y - y_center) ** 2
        for x in range(w):
            r = <int> round(sqrt(y2 + (x - x_center) ** 2))
            if im[y, x] > maxes[r]:
                maxes[r] = im[y, x]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def RadialAverage(np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] im,
                  np.ndarray[dtype=np.float32_t, ndim=1,
                             negative_indices=False, mode='c'] profile,
                  np.ndarray[dtype=np.float32_t, ndim=1,
                             negative_indices=False, mode='c'] counts,
                  int max_radius, int y_center, int x_center
                  ):
    cdef:
        int x, y, w, h, r, y2

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        y2 = (y - y_center) ** 2
        for x in range(w):
            r = <int> round(sqrt(y2 + (x - x_center) ** 2))
            if r > max_radius: continue
            profile[r] += im[y, x]
            counts[r] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def RadialAverage2(np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] im,
                   np.ndarray[dtype=np.float32_t, ndim=1,
                              negative_indices=False, mode='c'] profile,
                   np.ndarray[dtype=np.float32_t, ndim=1,
                              negative_indices=False, mode='c'] counts,
                   np.ndarray[dtype=np.int32_t, ndim=2,
                              negative_indices=False, mode='c'] dist
                   ):
    cdef:
        int x, y, w, h, r, y2

    h = im.shape[0]
    w = im.shape[1]

    for y in range(h):
        for x in range(w):
            profile[dist[y, x]] += im[y, x]
            counts[dist[y, x]] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ConnectToEdge(np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] boundaries,
                  int dist_to_edge):
    cdef:
        int h, w, x, y, nc, x2, y2

    h = boundaries.shape[0]
    w = boundaries.shape[1]

    # top
    for y in range(1, dist_to_edge):
        for x in range(1, w - 1):
            if boundaries[y, x] == 0: continue
            nc = 0
            if boundaries[y - 1, x - 1] == 1: nc += 1
            if boundaries[y - 1, x] == 1: nc += 1
            if boundaries[y - 1, x + 1] == 1: nc += 1
            if boundaries[y, x + 1] == 1: nc += 1
            if boundaries[y + 1, x + 1] == 1: nc += 1
            if boundaries[y + 1, x] == 1: nc += 1
            if boundaries[y + 1, x - 1] == 1: nc += 1
            if boundaries[y, x - 1] == 1: nc += 1
            if nc != 1: continue
            for y2 in range(0, y): boundaries[y2, x] = 1

    # left
    for x in range(1, dist_to_edge):
        for y in range(1, h - 1):
            if boundaries[y, x] == 0: continue
            nc = 0
            if boundaries[y - 1, x - 1] == 1: nc += 1
            if boundaries[y - 1, x] == 1: nc += 1
            if boundaries[y - 1, x + 1] == 1: nc += 1
            if boundaries[y, x + 1] == 1: nc += 1
            if boundaries[y + 1, x + 1] == 1: nc += 1
            if boundaries[y + 1, x] == 1: nc += 1
            if boundaries[y + 1, x - 1] == 1: nc += 1
            if boundaries[y, x - 1] == 1: nc += 1
            if nc != 1: continue
            for x2 in range(0, x): boundaries[y, x2] = 1

    # right
    for x in range(w - dist_to_edge, w - 1):
        for y in range(1, h - 1):
            if boundaries[y, x] == 0: continue
            nc = 0
            if boundaries[y - 1, x - 1] == 1: nc += 1
            if boundaries[y - 1, x] == 1: nc += 1
            if boundaries[y - 1, x + 1] == 1: nc += 1
            if boundaries[y, x + 1] == 1: nc += 1
            if boundaries[y + 1, x + 1] == 1: nc += 1
            if boundaries[y + 1, x] == 1: nc += 1
            if boundaries[y + 1, x - 1] == 1: nc += 1
            if boundaries[y, x - 1] == 1: nc += 1
            if nc != 1: continue
            for x2 in range(x, w): boundaries[y, x2] = 1

    # bottom
    for y in range(h - dist_to_edge, h - 1):
        for x in range(1, w - 1):
            if boundaries[y, x] == 0: continue

            nc = 0
            if boundaries[y - 1, x - 1] == 1: nc += 1
            if boundaries[y - 1, x] == 1: nc += 1
            if boundaries[y - 1, x + 1] == 1: nc += 1
            if boundaries[y, x + 1] == 1: nc += 1
            if boundaries[y + 1, x + 1] == 1: nc += 1
            if boundaries[y + 1, x] == 1: nc += 1
            if boundaries[y + 1, x - 1] == 1: nc += 1
            if boundaries[y, x - 1] == 1: nc += 1
            if nc != 1: continue

            for y2 in range(y, h): boundaries[y2, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def HillClimb1D(np.ndarray[dtype=np.float32_t, ndim=1,
                           negative_indices=False, mode='c'] signal,
                int start):
    cdef:
        int x, length

    length = signal.shape[0]
    if start == 0 or start == length - 1:
        return start

    x = start
    if signal[x] < signal[x + 1]:
        while x < length and signal[x] < signal[x + 1]:
            x += 1
    elif signal[x] < signal[x - 1]:
        while x > 0 and signal[x] < signal[x - 1]:
            x -= 1

    return x

@cython.boundscheck(False)
@cython.wraparound(False)
def FillGaps(np.ndarray[dtype=np.uint8_t, ndim=2,
                        negative_indices=False, mode='c'] mask):
    cdef:
        int h, w, x, y, nc, n1, skip, cn, i
        int locations[8]
    h = mask.shape[0]
    w = mask.shape[1]

    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if mask[y, x] == 1: continue

            # check num neighbours
            nc = 0
            if mask[y - 1, x - 1] == 1: nc += 1
            if mask[y - 1, x] == 1: nc += 1
            if mask[y - 1, x + 1] == 1: nc += 1
            if mask[y, x + 1] == 1: nc += 1
            if mask[y + 1, x + 1] == 1: nc += 1
            if mask[y + 1, x] == 1: nc += 1
            if mask[y + 1, x - 1] == 1: nc += 1
            if mask[y, x - 1] == 1: nc += 1
            if nc < 2 or nc > 4: continue

            # check crossing number
            locations[0] = mask[y - 1, x - 1]
            locations[1] = mask[y - 1, x]
            locations[2] = mask[y - 1, x + 1]
            locations[3] = mask[y, x + 1]
            locations[4] = mask[y + 1, x + 1]
            locations[5] = mask[y + 1, x]
            locations[6] = mask[y + 1, x - 1]
            locations[7] = mask[y, x - 1]
            cn = 0
            for i in range(8): cn += abs(locations[i] - locations[(i + 1) % 8])
            cn = cn // 2

            if cn < 2:
                continue
            elif cn == 2:
                # make sure no gaps of size 1
                skip = 0
                for n1 in range(8):
                    if (locations[n1] == 0 and
                                locations[(n1 - 1) % 8] == 1 and
                                locations[(n1 + 1) % 8] == 1): skip = 1
                if skip == 1: continue

            mask[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void bresenham(np.ndarray[dtype=np.uint8_t, ndim=2,
                               negative_indices=False, mode='c'] boundaries,
                    int y, int x, int y2, int x2):
    cdef:
        int steep = 0
        int dx = abs(x2 - x)
        int dy = abs(y2 - y)
        int sx, sy, d, i

    if (x2 - x) > 0:
        sx = 1
    else:
        sx = -1
    if (y2 - y) > 0:
        sy = 1
    else:
        sy = -1
    if dy > dx:
        steep = 1
        x, y = y, x
        dx, dy = dy, dx
        sx, sy = sy, sx
    d = (2 * dy) - dx

    for i in range(dx):
        if steep:
            boundaries[x, y] = 1
        else:
            boundaries[y, x] = 1
        while d >= 0:
            y = y + sy
            d = d - (2 * dx)
        x = x + sx
        d = d + (2 * dy)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int DislocationLine(np.ndarray[dtype=np.float32_t, ndim=2,
                                    negative_indices=False, mode='c'] im,
                         int y_start, int x_start, int y_end, int x_end,
                         np.ndarray[dtype=np.int32_t, ndim=2,
                                    negative_indices=False, mode='c'] coords):
    cdef:
        int dx, dy, err, e2
        int sx, sy, d, i, pixel_num, x, y
        np.float32_t start_val, prev_val
        int h, w

    h = im.shape[0]
    w = im.shape[1]
    start_val = im[y_start, x_start] * 0.99

    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)

    if x_start < x_end:
        sx = 1
    else:
        sx = -1
    if y_start < y_end:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    y = y_start
    x = x_start
    pixel_num = 0

    while True:
        if x < 0 or y < 0 or y >= h or x >= w: break

        coords[0, pixel_num] = y
        coords[1, pixel_num] = x
        pixel_num += 1

        if ((x == x_end and y == y_end) or
                (pixel_num > 1 and im[y, x] > start_val)): break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return pixel_num

#@cython.boundscheck(False)
#@cython.wraparound(False)
#def LineSumAA(params, np.ndarray[dtype=np.float32_t, ndim=2,
#                         negative_indices=False, mode='c'] im,
#              float x0, float x1):
#    cdef:
#        int xpxl1, xpxl2, x, steep, y, h
#        float dy, dx, gradient, xend, yend, intery, sum, w
#        float y0, y1
#
#    h = im.shape[0]
#    (y0, y1) = params
#    dx, dy = x1-x0, y1-y0
#    if abs(dy) > abs(dx): steep = 1
#    else: steep = 0
#
#    if steep == 1:
#        x0, x1, y0, y1, dy, dx = y0, y1, x0, x1, dx, dy
#    if x1 < x0:
#        x0, x1, y0, y1 = x1, x0, y1, y0
#
#    gradient = dy / dx
#    xpxl1 = <int>round(x0)
#    xpxl2 = <int>round(x1)
#    intery = y0 + (gradient * (round(x0) - x0)) + gradient
#
#    sum = 0
#
#    if steep == 0:
#        for x in range(xpxl1 + 1, xpxl2):
#            w = intery - floor(intery)
#            y = <int>intery
#            if y < 0 or y >= h: continue
#            sum += im[y,   x]*(1.0-w)
#            sum += im[y+1, x]*w
#            intery += gradient
#    else:
#        for x in range(xpxl1 + 1, xpxl2):
#            w = intery - floor(intery)
#            y = <int>intery
#            if y < 0 or y >= h: continue
#            sum += im[x, y  ]*(1.0-w)
#            sum += im[x, y+1]*w
#            intery += gradient
#
#    print y0, y1, sum
#
#    return -sum


@cython.boundscheck(False)
@cython.wraparound(False)
def LineDrawAA(np.ndarray[dtype=np.float32_t, ndim=2,
                          negative_indices=False, mode='c'] im,
               float y0, float x0, float y1, float x1):
    cdef:
        int xpxl1, xpxl2, x, steep
        float dy, dx, gradient, xend, yend, intery, sum, w

    dx, dy = x1 - x0, y1 - y0
    if abs(dy) > abs(dx):
        steep = 1
    else:
        steep = 0

    if steep == 1:
        x0, x1, y0, y1, dy, dx = y0, y1, x0, x1, dx, dy
    if x1 < x0:
        x0, x1, y0, y1 = x1, x0, y1, y0

    gradient = dy / dx
    xpxl1 = <int> round(x0)
    xpxl2 = <int> round(x1)
    intery = y0 + (gradient * (round(x0) - x0)) + gradient

    sum = 0

    if steep == 0:
        for x in range(xpxl1 + 1, xpxl2):
            w = intery - floor(intery)
            im[<int> intery, x] = 1.0 - (intery - floor(intery))
            im[<int> intery + 1, x] = (intery - floor(intery))
            intery += gradient
    else:
        for x in range(xpxl1 + 1, xpxl2):
            w = intery - floor(intery)
            im[x, <int> intery] = 1.0 - (intery - floor(intery))
            im[x, <int> intery + 1] = (intery - floor(intery))
            intery += gradient

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def LineSum(np.ndarray[dtype=np.float32_t, ndim=2,
                       negative_indices=False, mode='c'] im,
            int y_start, int x_start, int y_end, int x_end):
    cdef:
        int dx, dy, err, e2
        int sx, sy, d, i, pixel_num, x, y
        float start_val, prev_val, sum
        int h, w

    sum = 0

    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)

    if x_start < x_end:
        sx = 1
    else:
        sx = -1
    if y_start < y_end:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    y = y_start
    x = x_start
    pixel_num = 0

    while True:
        sum += im[y, x]
        pixel_num += 1

        if (x == x_end and y == y_end): break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return sum

@cython.boundscheck(False)
@cython.wraparound(False)
def LineVals(np.ndarray[dtype=np.float32_t, ndim=2,
                        negative_indices=False, mode='c'] im,
             np.ndarray[dtype=np.float32_t, ndim=1,
                        negative_indices=False, mode='c'] vals,
             int y_start, int x_start, int y_end, int x_end):
    cdef:
        int dx, dy, err, e2
        int sx, sy, d, i, pixel_num, x, y
        float start_val, prev_val, sum
        int h, w


    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)

    if x_start < x_end:
        sx = 1
    else:
        sx = -1
    if y_start < y_end:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    y = y_start
    x = x_start
    pixel_num = 0

    while True:
        vals[pixel_num] = im[y, x]

        pixel_num += 1

        if (x == x_end and y == y_end): break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return pixel_num

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int LineCoords(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    int y_start, int x_start, int y_end, int x_end,
                    np.ndarray[dtype=np.int32_t, ndim=2,
                               negative_indices=False, mode='c'] coords):
    cdef:
        int dx, dy, err, e2
        int sx, sy, d, i, pixel_num, x, y
        np.float32_t start_val, prev_val
        int h, w

    h = im.shape[0]
    w = im.shape[1]

    dx = abs(x_end - x_start)
    dy = abs(y_end - y_start)

    if x_start < x_end:
        sx = 1
    else:
        sx = -1
    if y_start < y_end:
        sy = 1
    else:
        sy = -1
    err = dx - dy
    y = y_start
    x = x_start
    pixel_num = 0

    while True:
        if x < 0 or y < 0 or y >= h or x >= w: break

        coords[0, pixel_num] = y
        coords[1, pixel_num] = x
        pixel_num += 1

        if (x == x_end and y == y_end): break

        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return pixel_num

@cython.boundscheck(False)
@cython.wraparound(False)
def DislocationMask(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] edges,
                    np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] edgeH,
                    np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] edgeV,
                    np.ndarray[dtype=np.uint8_t, ndim=2,
                               negative_indices=False, mode='c'] dis_mask,
                    np.float32_t edge_thresh,
                    np.ndarray[dtype=np.int32_t, ndim=2,
                               negative_indices=False, mode='c'] coords
                    ):
    cdef:
        np.int32_t x, y, w, h, x2, y2, max_d, num_pixels, i
        np.float32_t theta, length, diff


    h = im.shape[0]
    w = im.shape[1]
    max_d = coords.shape[1] - 1

    for y in range(h):
        for x in range(w):
            if edges[y, x] <= edge_thresh:
                continue
            if im[y, x] < 0.2:
                continue

            theta = atan2(-edgeH[y, x], -edgeV[y, x])

            x2 = x + <int> round(cos(theta) * max_d)
            y2 = y + <int> round(sin(theta) * max_d)

            num_pixels = DislocationLine(im, y, x, y2, x2, coords)
            for i in range(num_pixels):
                dis_mask[coords[0, i], coords[1, i]] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FillMasks(np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] background,
              np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] dislocation_mask,
              np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] dislocations_filled,
              int downsize,
              np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] im_filtered,
              ):
    cdef:
        np.int32_t x, y, w, h, found_tran

    h = background.shape[0]
    w = background.shape[1]
    for y in range(h):
        for x in range(w):
            if dislocation_mask[y, x] == 1:
                background[y, x] = dislocations_filled[y / downsize, x / downsize]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def InterpolateDislocations(np.ndarray[dtype=np.float32_t, ndim=2,
                                       negative_indices=False, mode='c'] im,
                            np.ndarray[dtype=np.uint8_t, ndim=2,
                                       negative_indices=False, mode='c'] mask,
                            np.ndarray[dtype=np.float32_t, ndim=2,
                                       negative_indices=False, mode='c'] results,
                            np.ndarray[dtype=np.float32_t, ndim=2,
                                       negative_indices=False, mode='c'] surface,
                            #np.ndarray[dtype=np.uint8_t, ndim=2,
                            #          negative_indices=False, mode='c'] impure_area,
                            np.ndarray[dtype=np.float32_t, ndim=1,
                                       negative_indices=False, mode='c'] scratch,
                            np.ndarray[dtype=np.int32_t, ndim=2,
                                       negative_indices=False, mode='c'] coords):
    cdef:
        np.int32_t x, y, w, h, x1, y1, x2, y2
        int num_vals, radius
        int i, j, l, r, mid_i
        float mid_val, tmp, val
        int left, right, top, bottom, ww, wh
        int target_vals, num_coords, tran_found

    h = im.shape[0]
    w = im.shape[1]
    target_vals = scratch.shape[0]
    num_coords = coords.shape[1]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0: continue

            num_vals = 0
            for r in range(num_coords):
                j = y + coords[0, r]
                i = x + coords[1, r]

                if (j >= 0 and j < h and i >= 0 and i < w and mask[j, i] == 0):
                    scratch[num_vals] = im[j, i]
                    num_vals = num_vals + 1

                if num_vals >= target_vals: break

            # if we didn't find enough unmasked pixels, use the surface fit
            if num_vals < 40:
                results[y, x] = surface[y, x]
                continue

            # find median value
            l = 0
            r = num_vals - 1
            mid_i = num_vals / 2
            while l < r:
                mid_val = scratch[mid_i]
                i = l
                j = r
                while 1:
                    while scratch[i] < mid_val: i = i + 1
                    while mid_val < scratch[j]: j = j - 1
                    if i <= j:
                        tmp = scratch[i]
                        scratch[i] = scratch[j]
                        scratch[j] = tmp
                        i = i + 1
                        j = j - 1
                    if i > j: break
                if j < mid_i: l = i
                if mid_i < i: r = j
            results[y, x] = scratch[mid_i]

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def PitTexture(np.ndarray[dtype=np.float32_t, ndim=2,
                          negative_indices=False, mode='c'] im,
               np.ndarray[dtype=np.float32_t, ndim=2,
                          negative_indices=False, mode='c'] dips,
               int t):
    cdef:
        int x, y, width, height, x2, y2, i
        float m
        float n, ne, e, se, s, sw, w, nw

    height, width = im.shape[0], im.shape[1]

    for y in range(t, height - t - 1):
        for x in range(t, width - t - 1):
            #m = 1000
            #for y2 in range(y-s, y+s+1):
            #    m = min(m, im[y2, x-s], im[y2, x+s+1])
            #    if m < im[y, x]: break
            #for x2 in range(x-s, x+s+1):
            #    m = min(m, im[y-s, x2], im[y+s+1, x2])
            #    if m < im[y, x]: break
            #
            n, ne, e, se, s, sw, w, nw = 0, 0, 0, 0, 0, 0, 0, 0
            for i in range(1, t):
                if im[y - i, x] > n: n = im[y - i, x]
                if im[y - i, x + i] > n: ne = im[y - i, x + i]
                if im[y, x + i] > n: e = im[y, x + i]
                if im[y + i, x + i] > n: se = im[y + i, x + i]
                if im[y + i, x] > n: s = im[y + i, x]
                if im[y + i, x - i] > n: sw = im[y + i, x - i]
                if im[y, x - i] > n: w = im[y, x - i]
                if im[y - i, x - i] > n: nw = im[y - i, x - i]
            m = min(n, ne, e, se, s, sw, w, nw)
            if m > im[y, x]:
                dips[y, x] = m - im[y, x]

@cython.boundscheck(False)
@cython.wraparound(False)
def FillCorners(np.ndarray[dtype=np.float32_t, ndim=2,
                           negative_indices=False, mode='c'] im,
                np.ndarray[dtype=np.int32_t, ndim=1,
                           negative_indices=False, mode='c'] ys,
                np.ndarray[dtype=np.int32_t, ndim=1,
                           negative_indices=False, mode='c'] xs):
    cdef:
        int h, w, y, x, w2, r, c

    h = im.shape[0]
    w = im.shape[1]
    w2 = w // 2
    for (y, c) in zip(ys, xs):
        if c < w2:
            for x in range(c - 1, -1, -1): im[y, x] = im[y, c]
        else:
            for x in range(c + 1, w): im[y, x] = im[y, c]

@cython.boundscheck(False)
@cython.wraparound(False)
def MakeMonotonic(np.ndarray[dtype=np.float32_t, ndim=1,
                             negative_indices=False, mode='c'] signal_in,
                  np.ndarray[dtype=np.float32_t, ndim=1,
                             negative_indices=False, mode='c'] signal_out):
    cdef:
        int x, length

    length = signal_in.shape[0]
    signal_out[0] = signal_in[0]
    for x in range(1, length):
        signal_out[x] = max(signal_out[x - 1], signal_in[x])

@cython.boundscheck(False)
@cython.wraparound(False)
def MakeMonotonicBBs(np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] im_in,
                     np.ndarray[dtype=np.float32_t, ndim=2,
                                negative_indices=False, mode='c'] im_out,
                     np.ndarray[dtype=np.int32_t, ndim=1,
                                negative_indices=False, mode='c'] bb_locs):
    cdef:
        int h, w, y, x, bb

    h = im_in.shape[0]
    w = im_in.shape[1]
    for y in range(h):
        bb = 0
        im_out[y, 0] = im_in[y, 0]
        for x in range(1, w):
            if x == bb_locs[bb]:
                if bb < bb_locs.shape[0] - 1:
                    bb += 1
                im_out[y, x] = im_in[y, x]
            else:
                im_out[y, x] = max(im_out[y, x - 1], im_in[y, x])

@cython.boundscheck(False)
@cython.wraparound(False)
def ExpandFingers(np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] result,
                  np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] im_small,
                  np.ndarray[dtype=np.int32_t, ndim=1,
                             negative_indices=False, mode='c'] rows):
    cdef:
        int h, w, y, x, bb, r, num_rows, row1, row2
        float v1, v2, v, d, s, steps

    num_rows = rows.shape[0]
    h = result.shape[0]
    w = result.shape[1]

    for x in range(w):
        # top
        for y in range(rows[0]):
            result[y, x] = im_small[0, x]

        # bottom
        for y in range(rows[num_rows - 1], h):
            result[y, x] = im_small[num_rows - 1, x]

    # main
    for r in range(num_rows - 1):
        row1 = rows[r]
        row2 = rows[r + 1]
        steps = row2 + 1 - row1
        for x in range(w):
            v1 = im_small[r, x]
            v2 = im_small[r + 1, x]
            d = v2 - v1
            s = 0
            for y in range(row1, row2 + 1):
                result[y, x] = (s / steps) * v2 + ((steps - s) / steps) * v1
                s += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def InterpolateFingers(np.ndarray[dtype=np.float32_t, ndim=2,
                                  negative_indices=False, mode='c'] im,
                       np.ndarray[dtype=np.float32_t, ndim=2,
                                  negative_indices=False, mode='c'] result,
                       np.ndarray[dtype=np.int32_t, ndim=1,
                                  negative_indices=False, mode='c'] rows):
    cdef:
        int h, w, y, x, bb, r, num_rows, row1, row2
        float v1, v2, v, d, s, steps

    num_rows = rows.shape[0]
    h = im.shape[0]
    w = im.shape[1]

    for x in range(w):
        # top
        for y in range(rows[0]):
            result[y, x] = im[rows[0], x]

        # bottom
        for y in range(rows[num_rows - 1], h):
            result[y, x] = im[rows[num_rows - 1], x]

    # main
    for r in range(num_rows - 1):
        row1 = rows[r]
        row2 = rows[r + 1]
        steps = row2 + 1 - row1
        for x in range(w):
            v1 = im[row1, x]
            v2 = im[row2, x]
            d = v2 - v1
            s = 0
            for y in range(row1, row2 + 1):
                result[y, x] = (s / steps) * v2 + ((steps - s) / steps) * v1
                s += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def DarkSpots(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] im,
              np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] strength,
              int s):
    cdef:
        int h, w, y, x, i, j, dy, dx
        float d1, d2, d3, d4, d5, d6, d7, d8, max1, max2, v

    h, w = im.shape[0], im.shape[1]

    for y in range(s, h - s):
        for x in range(s, w - s):
            d1, d2, d3, d4, d5, d6, d7, d8 = 0, 0, 0, 0, 0, 0, 0, 0,
            for i in range(1, s + 1):
                # |
                dx, dy = 0, 1
                v = im[y + (i * dy), x + (i * dx)]
                if v > d1: d1 = v
                v = im[y + (i * dy * -1), x + (i * dx * -1)]
                if v > d2: d2 = v

                # \
                dx, dy = 1, 1
                v = im[y + (i * dy), x + (i * dx)]
                if v > d3: d3 = v
                v = im[y + (i * dy * -1), x + (i * dx * -1)]
                if v > d4: d4 = v

                # -
                dx, dy = 1, 0
                v = im[y + (i * dy), x + (i * dx)]
                if v > d5: d5 = v
                v = im[y + (i * dy * -1), x + (i * dx * -1)]
                if v > d6: d6 = v

                # /
                dx, dy = 1, -1
                v = im[y + (i * dy), x + (i * dx)]
                if v > d7: d7 = v
                v = im[y + (i * dy * -1), x + (i * dx * -1)]
                if v > d8: d8 = v

            v = min(d1, d2, d3, d4, d5, d6, d7, d8) - im[y, x]
            strength[y, x] = max(v, 0)

            # d1, d2, d3, d4 = 0, 0, 0, 0
            # |
            # if im[y, x] < im[y-1, x] and im[y, x] < im[y+1, x]:
            #     dx, dy = 0, 1
            #     max1, max2 = 0, 0
            #     for i in range(1, s+1):
            #         v = im[y+(i*dy), x+(i*dx)]
            #         if v > max1: max1 = v
            #         v = im[y+(i*dy*-1), x+(i*dx*-1)]
            #         if v > max2: max2 = v
            #     d1 = min(max1, max2) - im[y, x]
            #
            # # \
            # if im[y, x] < im[y+1, x+1] and im[y, x] < im[y-1, x-1]:
            #     dx, dy = 1, 1
            #     max1, max2 = 0, 0
            #     for i in range(1, s+1):
            #         v = im[y+(i*dy), x+(i*dx)]
            #         if v > max1: max1 = v
            #         v = im[y+(i*dy*-1), x+(i*dx*-1)]
            #         if v > max2: max2 = v
            #     d2 = min(max1, max2) - im[y, x]
            #
            # # -
            # if im[y, x] < im[y, x+1] and im[y, x] < im[y, x-1]:
            #     dx, dy = 1, 0
            #     max1, max2 = 0, 0
            #     for i in range(1, s+1):
            #         v = im[y+(i*dy), x+(i*dx)]
            #         if v > max1: max1 = v
            #         v = im[y+(i*dy*-1), x+(i*dx*-1)]
            #         if v > max2: max2 = v
            #     d3 = min(max1, max2) - im[y, x]
            #
            # # /
            # if im[y, x] < im[y-1, x+1] and im[y, x] < im[y+1, x-1]:
            #     dx, dy = 1, -1
            #     max1, max2 = 0, 0
            #     for i in range(1, s+1):
            #         v = im[y+(i*dy), x+(i*dx)]
            #         if v > max1: max1 = v
            #         v = im[y+(i*dy*-1), x+(i*dx*-1)]
            #         if v > max2: max2 = v
            #     d4 = min(max1, max2) - im[y, x]
            #
            # strength[y, x] = max(d1, d2, d3, d4)

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def BrokenFingerBrightLines(np.ndarray[dtype=np.float32_t, ndim=2,
                                       negative_indices=False, mode='c'] finger_im,
                            np.ndarray[dtype=np.int32_t, ndim=1,
                                       negative_indices=False, mode='c'] bb_locations,
                            np.ndarray[dtype=np.float32_t, ndim=2,
                                       negative_indices=False, mode='c'] break_strength,
                            np.ndarray[dtype=np.uint8_t, ndim=2,
                                       negative_indices=False, mode='c'] breaks,
                            np.ndarray[dtype=np.float32_t, ndim=2,
                                       negative_indices=False, mode='c'] bright_lines):
    cdef:
        int h, w, y, x, bb, break_count, i
        float max1, max2, bright_low, bright_high, v1, v2, bright_range
        int max1_i, max2_i, inc, num_lines

    h, w = finger_im.shape[0], finger_im.shape[1]
    num_lines = 0
    for i in range(bb_locations.shape[0] - 1):
        for y in range(h):
            break_count = 0
            for x in range(bb_locations[i], bb_locations[i + 1]):
                if breaks[y, x] == 1:
                    break_count += 1
            if break_count == 0: continue

            num_lines += 1
            max1, max2 = -1, -1
            max1_i, max2_i = -1, -1
            if break_count > 2:
                # only keep 2 strongest

                # find strongest
                for x in range(bb_locations[i], bb_locations[i + 1]):
                    if breaks[y, x] == 1 and break_strength[y, x] > max1:
                        max1 = break_strength[y, x]
                        max1_i = x
                # find second strongest
                for x in range(bb_locations[i], bb_locations[i + 1]):
                    if breaks[y, x] == 1 and break_strength[y, x] > max2 and break_strength[y, x] < max1:
                        max2 = break_strength[y, x]
                        max2_i = x
                # remove the others
                for x in range(bb_locations[i], bb_locations[i + 1]):
                    if breaks[y, x] == 1 and x != max1_i and x != max2_i:
                        breaks[y, x] = 0
            else:
                # find strongest
                for x in range(bb_locations[i], bb_locations[i + 1]):
                    if breaks[y, x] == 1 and break_strength[y, x] > max1:
                        max1 = break_strength[y, x]
                        max1_i = x

            if max1_i < 10 or max1_i > w - 10: continue

            # highlight bright lines
            # find brightest and darkest points in the neighbourhood of break
            v1 = finger_im[y, max1_i - 10]
            v2 = finger_im[y, max1_i + 10]
            if v1 > v2:
                inc, stop = -1, bb_locations[i]
            else:
                inc, stop = 1, bb_locations[i + 1]

            bright_high = 0
            for x in range(max1_i, max1_i + (20 * inc), inc):
                if finger_im[y, x] > bright_high: bright_high = finger_im[y, x]
            bright_low = 10
            for x in range(max1_i, max1_i + (20 * inc * -1), inc * -1):
                if finger_im[y, x] < bright_low: bright_low = finger_im[y, x]

            bright_range = bright_high - bright_low
            for x in range(max1_i + inc, stop, inc):
                bright_lines[y, x] = (finger_im[y, x] - bright_low) / bright_range

                # stop if we reach another break
                if breaks[y, x + inc] == 1: break

    return num_lines

@cython.boundscheck(False)
@cython.wraparound(False)
def InterpolateBBs(np.ndarray[dtype=np.float32_t, ndim=2,
                              negative_indices=False, mode='c'] im,
                   np.ndarray[dtype=np.int32_t, ndim=1,
                              negative_indices=False, mode='c'] locations,
                   int bb_width):
    cdef:
        int h, w, y, x, bb
        float v1, v2, v, d, s, steps

    h = im.shape[0]
    w = im.shape[1]

    steps = (2 * bb_width) + 1
    for bb in locations:
        for y in range(h):
            v1 = im[y, bb - bb_width]
            v2 = im[y, bb + bb_width + 1]
            d = v2 - v1
            s = 0
            for x in range(bb - bb_width, bb + bb_width + 1):
                im[y, x] = (s / steps) * v2 + ((steps - s) / steps) * v1
                s += 1

@cython.boundscheck(False)
@cython.wraparound(False)
def ConnectBroken(np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] boundaries,
                  np.ndarray[dtype=np.float32_t, ndim=2,
                             negative_indices=False, mode='c'] gradient,
                  int max_dist):
    """
    Neighbour labels:
        0 1 2
        7 x 3
        6 5 4
    """
    cdef:
        unsigned short x, y, x2, y2, x_start, x_stop, y_start, y_stop, w, h
        unsigned short closest_x, closest_y, closest_dist
        unsigned int p, num_pixels
        unsigned char nc, src
        unsigned char neighbourhood[9][9]
        double dist, closest_d

    h = boundaries.shape[0]
    w = boundaries.shape[1]

    # first try to connect two endpoints
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if boundaries[y, x] == 0:
                continue

            # count neighbours
            nc = 0

            if boundaries[y - 1, x - 1] == 1:
                nc += 1
                src = 0
            if boundaries[y - 1, x] == 1:
                nc += 1
                src = 1
            if boundaries[y - 1, x + 1] == 1:
                nc += 1
                src = 2
            if boundaries[y, x + 1] == 1:
                nc += 1
                src = 3
            if boundaries[y + 1, x + 1] == 1:
                nc += 1
                src = 4
            if boundaries[y + 1, x] == 1:
                nc += 1
                src = 5
            if boundaries[y + 1, x - 1] == 1:
                nc += 1
                src = 6
            if boundaries[y, x - 1] == 1:
                nc += 1
                src = 7

            if nc != 1:
                continue

            # attempt # 1 - follow maximum lines in gradient to see if any other
            #  pixel is found
            if AttemptConnect(boundaries, gradient, y, x, src,
                              max_dist, max_dist, False) == 1:
                continue

    return

@cython.boundscheck(False)
@cython.wraparound(False)
cdef unsigned char NumNeighbours(np.ndarray[dtype=np.uint8_t, ndim=2,
                                            negative_indices=False, mode='c'] boundaries,
                                 unsigned short y,
                                 unsigned short x):
    cdef:
        unsigned char nc

    if ((y == 0 or y == boundaries.shape[0] - 1) or
            (x == 0 or x == boundaries.shape[1] - 1)): return 0

    nc = 0

    if boundaries[y - 1, x - 1] == 1: nc += 1
    if boundaries[y - 1, x] == 1: nc += 1
    if boundaries[y - 1, x + 1] == 1: nc += 1
    if boundaries[y, x + 1] == 1: nc += 1
    if boundaries[y + 1, x + 1] == 1: nc += 1
    if boundaries[y + 1, x] == 1: nc += 1
    if boundaries[y + 1, x - 1] == 1: nc += 1
    if boundaries[y, x - 1] == 1: nc += 1

    return nc

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int AttemptConnect(np.ndarray[dtype=np.uint8_t, ndim=2,
                                   negative_indices=False, mode='c'] boundaries,
                        np.ndarray[dtype=np.float32_t, ndim=2,
                                   negative_indices=False, mode='c'] gradient,
                        unsigned short y,
                        unsigned short x,
                        unsigned short src,
                        int level,
                        int max_level,
                        debug
                        ):
    cdef:
        unsigned short ns[8][2], check[5]
        unsigned int n, num_check
        int max_n
        float max_g

    if level < 0:
        return 0

    if ((y == 0 or y == boundaries.shape[0] - 1) or
            (x == 0 or x == boundaries.shape[1] - 1)): return 0

    ns[0][0] = x - 1
    ns[1][0] = x
    ns[2][0] = x + 1
    ns[3][0] = x + 1
    ns[4][0] = x + 1
    ns[5][0] = x
    ns[6][0] = x - 1
    ns[7][0] = x - 1
    ns[0][1] = y - 1
    ns[1][1] = y - 1
    ns[2][1] = y - 1
    ns[3][1] = y
    ns[4][1] = y + 1
    ns[5][1] = y + 1
    ns[6][1] = y + 1
    ns[7][1] = y

    check[0] = (src + 4) % 8
    check[1] = (src + 3) % 8
    check[2] = (src + 5) % 8
    check[3] = (src + 2) % 8
    check[4] = (src + 6) % 8

    # check if any of the candidate neighbours are 1. if so, we're done, so
    #  recurse back up the stack
    for n in range(5):
        if boundaries[ns[check[n]][1], ns[check[n]][0]] == 1:
            boundaries[y, x] = 1

            # check if this was an isolated point. if so, recharge and keep going
            if NumNeighbours(boundaries, ns[check[n]][1],
                             ns[check[n]][0]) == 1:
                AttemptConnect(boundaries, gradient,
                               ns[check[n]][1],
                               ns[check[n]][0],
                               (check[n] + 4) % 8,
                               max_level, max_level, debug)

            return 1

    # move on to the maximum of candidates
    max_g = 0
    max_n = -1
    for n in range(3):
        if gradient[ns[check[n]][1], ns[check[n]][0]] > max_g:
            max_g = gradient[ns[check[n]][1], ns[check[n]][0]]
            max_n = n

    if AttemptConnect(boundaries, gradient,
                      ns[check[max_n]][1],
                      ns[check[max_n]][0],
                      (check[max_n] + 4) % 8,
                      level - 1,
                      max_level,
                      debug
                      ) == 1:
        boundaries[y, x] = 1
        return 1
    else:
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def BusBarsIncreasing(np.ndarray[dtype=np.float32_t, ndim=2,
                                 negative_indices=False, mode='c'] bb_in,
                      np.ndarray[dtype=np.float32_t, ndim=2,
                                 negative_indices=False, mode='c'] bb_out):
    cdef:
        int x, y, w, h, min_i
        float min_v

    h, w = bb_in.shape[0], bb_in.shape[1]
    for y in range(h):
        min_v = 10000
        for x in range(w):
            if bb_in[y, x] < min_v:
                min_v = bb_in[y, x]
                min_i = x

        bb_out[y, min_i] = bb_in[y, min_i]
        for x in range(min_i + 1, w):
            bb_out[y, x] = max(bb_in[y, x], bb_out[y, x - 1])
        for x in range(min_i - 1, -1, -1):
            bb_out[y, x] = max(bb_in[y, x], bb_out[y, x + 1])

@cython.boundscheck(False)
@cython.wraparound(False)
def FillHoles(np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] boundaries):
    cdef:
        unsigned int x, y, w, h
        unsigned char nc

    h = boundaries.shape[0] - 2
    w = boundaries.shape[1] - 2

    for y in range(2, h):
        for x in range(2, w):
            if boundaries[y, x] == 1:
                continue

            # count neighbours
            nc = 0
            if boundaries[y - 1, x - 1] == 1: nc += 1
            if boundaries[y - 1, x] == 1: nc += 1
            if boundaries[y - 1, x + 1] == 1: nc += 1
            if boundaries[y, x + 1] == 1: nc += 1
            if boundaries[y + 1, x + 1] == 1: nc += 1
            if boundaries[y + 1, x] == 1: nc += 1
            if boundaries[y + 1, x - 1] == 1: nc += 1
            if boundaries[y, x - 1] == 1: nc += 1

            if nc >= 6:
                boundaries[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def FastThin(np.ndarray[dtype=np.uint8_t, ndim=2,
                        negative_indices=False, mode='c'] boundaries,
             np.ndarray[dtype=np.int32_t, ndim=1,
                        negative_indices=False, mode='c'] ys,
             np.ndarray[dtype=np.int32_t, ndim=1,
                        negative_indices=False, mode='c'] xs,
             np.ndarray[dtype=np.uint8_t, ndim=1,
                        negative_indices=False, mode='c'] lut
             ):
    cdef:
        unsigned short x, y
        unsigned char code, pixelRemoved
        unsigned int i, num_pixels
        int h, w

    h = boundaries.shape[0]
    w = boundaries.shape[1]

    with nogil:
        num_pixels = <unsigned int> ys.shape[0]

        # thinning
        pixelRemoved = 1
        while pixelRemoved == 1:
            pixelRemoved = 0

            # pass 1 - remove the 1's and 3's
            for i in range(num_pixels):
                if ys[i] == -1:
                    continue
                if boundaries[ys[i], xs[i]] == 0:
                    ys[i] = -1
                    xs[i] = -1
                    continue

                y = ys[i]
                x = xs[i]
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    continue
                code = lut[(boundaries[y - 1, x - 1] * 1 +
                            boundaries[y - 1, x] * 2 +
                            boundaries[y - 1, x + 1] * 4 +
                            boundaries[y, x + 1] * 8 +
                            boundaries[y + 1, x + 1] * 16 +
                            boundaries[y + 1, x] * 32 +
                            boundaries[y + 1, x - 1] * 64 +
                            boundaries[y, x - 1] * 128)]

                if code == 1 or code == 3:
                    boundaries[y, x] = 0
                    pixelRemoved = 1

            # pass 2 - remove the 2's and 3's
            for i in range(num_pixels):
                if ys[i] == -1:
                    continue
                if boundaries[ys[i], xs[i]] == 0:
                    ys[i] = -1
                    xs[i] = -1
                    continue

                y = ys[i]
                x = xs[i]
                if x == 0 or x == w - 1 or y == 0 or y == h - 1:
                    continue
                code = lut[(boundaries[y - 1, x - 1] * 1 +
                            boundaries[y - 1, x] * 2 +
                            boundaries[y - 1, x + 1] * 4 +
                            boundaries[y, x + 1] * 8 +
                            boundaries[y + 1, x + 1] * 16 +
                            boundaries[y + 1, x] * 32 +
                            boundaries[y + 1, x - 1] * 64 +
                            boundaries[y, x - 1] * 128)]

                if code == 2 or code == 3:
                    boundaries[y, x] = 0
                    pixelRemoved = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def LocalMins(np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] im,
              np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] local_mins):
    cdef:
        np.int32_t x, y, w, h

    h = im.shape[0]
    w = im.shape[1]

    for y in range(1, h - 1, 1):
        for x in range(1, w - 1, 1):
            if ((im[y, x] < im[y - 1, x] and im[y, x] < im[y + 1, x]) or
                    (im[y, x] < im[y, x - 1] and im[y, x] < im[y, x + 1]) or
                    (im[y, x] < im[y - 1, x - 1] and im[y, x] < im[y + 1, x + 1]) or
                    (im[y, x] < im[y + 1, x - 1] and im[y, x] < im[y - 1, x + 1])):
                local_mins[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def LocalMaxs(np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] im,
              np.ndarray[dtype=np.uint8_t, ndim=2,
                         negative_indices=False, mode='c'] local_maxs):
    cdef:
        np.int32_t x, y, w, h

    h = im.shape[0]
    w = im.shape[1]

    for y in range(1, h - 1, 1):
        for x in range(1, w - 1, 1):
            if (im[y, x] > im[y - 1, x - 1] and im[y, x] > im[y - 1, x] and
                        im[y, x] > im[y - 1, x + 1] and im[y, x] > im[y, x + 1] and
                        im[y, x] > im[y + 1, x + 1] and im[y, x] > im[y + 1, x] and
                        im[y, x] > im[y + 1, x - 1] and im[y, x] > im[y, x - 1]):
                local_maxs[y, x] = 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def GradientProfile(np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] im,
                    np.ndarray[dtype=np.float32_t, ndim=2,
                               negative_indices=False, mode='c'] edges,
                    np.ndarray[dtype=np.float32_t, ndim=1,
                               negative_indices=False, mode='c'] profile,
                    float edge_thresh):
    cdef:
        int x, y, w, h, count, s, s2, i, j, max_i
        float g, max_g
        int ys[8], xs[8]

    h = im.shape[0]
    w = im.shape[1]
    s = profile.shape[0]
    s2 = (profile.shape[0] - 1) // 2
    count = 0
    ys[0], ys[1], ys[2], ys[3], ys[4], ys[5], ys[6], ys[7] = -1, -1, -1, 0, 1, 1, 1, 0
    xs[0], xs[1], xs[2], xs[3], xs[4], xs[5], xs[6], xs[7] = -1, 0, 1, 1, 1, 0, -1, -1

    for y in range(s2, h - s2 - 1, 1):
        for x in range(s2, w - s2 - 1, 1):
            if edges[y, x] < edge_thresh: continue

            count += 1
            max_g = 0
            for i in range(8):
                g = im[y + ys[(i + 4) % 8], x + xs[(i + 4) % 8]] - im[y + ys[i], x + xs[i]]
                if g > max_g: max_g, max_i = g, i

            if max_i == 0:
                for i in range(s): profile[i] += im[y + s2 - i, x + s2 - i] - im[y, x]
            elif max_i == 1:
                for i in range(s):
                    profile[i] += im[y + s2 - i, x] - im[y, x]
            elif max_i == 2:
                for i in range(s): profile[i] += im[y + s2 - i, x - s2 + i] - im[y, x]
            elif max_i == 3:
                for i in range(s): profile[i] += im[y, x - s2 + i] - im[y, x]
            elif max_i == 4:
                for i in range(s): profile[i] += im[y - s2 + i, x - s2 + i] - im[y, x]
            elif max_i == 5:
                for i in range(s): profile[i] += im[y - s2 + i, x] - im[y, x]
            elif max_i == 6:
                for i in range(s): profile[i] += im[y - s2 + i, x + s2 - i] - im[y, x]
            elif max_i == 7:
                for i in range(s): profile[i] += im[y, x + s2 - i] - im[y, x]

    return count

@cython.boundscheck(False)
@cython.wraparound(False)
def RemoveSpur(np.ndarray[dtype=np.uint8_t, ndim=2,
                          negative_indices=False, mode='c'] mask,
               int y_prev, int x_prev, int y, int x, int depth, int max_depth):
    cdef:
        int ret, cn, vals[8], i

    # no junction found
    if depth == max_depth:
        return 0

    # count neighbours
    vals[0] = mask[y - 1, x - 1]
    vals[1] = mask[y - 1, x]
    vals[2] = mask[y - 1, x + 1]
    vals[3] = mask[y, x + 1]
    vals[4] = mask[y + 1, x + 1]
    vals[5] = mask[y + 1, x]
    vals[6] = mask[y + 1, x - 1]
    vals[7] = mask[y, x - 1]
    cn = 0
    for i in range(8):
        cn += abs(vals[i] - vals[(i + 1) % 8])
    cn = cn // 2

    if cn < 2:
        # end of short ridge found
        mask[y, x] = 0
        return 1
    elif cn > 2:
        return 1

    if mask[y, x - 1] == 1 and (y_prev != y or x_prev != x - 1):
        ret = RemoveSpur(mask, y, x, y, x - 1, depth + 1, max_depth)
    elif mask[y - 1, x] == 1 and (y_prev != y - 1 or x_prev != x):
        ret = RemoveSpur(mask, y, x, y - 1, x, depth + 1, max_depth)
    elif mask[y, x + 1] == 1 and (y_prev != y or x_prev != x + 1):
        ret = RemoveSpur(mask, y, x, y, x + 1, depth + 1, max_depth)
    elif mask[y + 1, x] == 1 and (y_prev != y + 1 or x_prev != x):
        ret = RemoveSpur(mask, y, x, y + 1, x, depth + 1, max_depth)
    elif mask[y - 1, x + 1] == 1 and (y_prev != y - 1 or x_prev != x + 1):
        ret = RemoveSpur(mask, y, x, y - 1, x + 1, depth + 1, max_depth)
    elif mask[y + 1, x + 1] == 1 and (y_prev != y + 1 or x_prev != x + 1):
        ret = RemoveSpur(mask, y, x, y + 1, x + 1, depth + 1, max_depth)
    elif mask[y + 1, x - 1] == 1 and (y_prev != y + 1 or x_prev != x - 1):
        ret = RemoveSpur(mask, y, x, y + 1, x - 1, depth + 1, max_depth)
    elif mask[y - 1, x - 1] == 1 and (y_prev != y - 1 or x_prev != x - 1):
        ret = RemoveSpur(mask, y, x, y - 1, x - 1, depth + 1, max_depth)

    if ret == 0:
        return 0
    else:
        mask[y, x] = 0
        return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def RemoveSpurs(np.ndarray[dtype=np.uint8_t, ndim=2,
                           negative_indices=False, mode='c'] mask,
                int max_depth):
    cdef:
        int h, w, x, y, ret, cn, vals[8], i

    h = mask.shape[0]
    w = mask.shape[1]

    for y in range(1, h - 1, 1):
        for x in range(1, w - 1, 1):
            if mask[y, x] == 0:
                continue

            # count neighbours
            vals[0] = mask[y - 1, x - 1]
            vals[1] = mask[y - 1, x]
            vals[2] = mask[y - 1, x + 1]
            vals[3] = mask[y, x + 1]
            vals[4] = mask[y + 1, x + 1]
            vals[5] = mask[y + 1, x]
            vals[6] = mask[y + 1, x - 1]
            vals[7] = mask[y, x - 1]
            cn = 0
            for i in range(8):
                cn += abs(vals[i] - vals[(i + 1) % 8])
            cn = cn // 2

            # only consider end points
            if cn != 1: continue

            if mask[y, x - 1] == 1:
                ret = RemoveSpur(mask, y, x, y, x - 1, 0, max_depth)
            elif mask[y - 1, x] == 1:
                ret = RemoveSpur(mask, y, x, y - 1, x, 0, max_depth)
            elif mask[y, x + 1] == 1:
                ret = RemoveSpur(mask, y, x, y, x + 1, 0, max_depth)
            elif mask[y + 1, x] == 1:
                ret = RemoveSpur(mask, y, x, y + 1, x, 0, max_depth)
            elif mask[y - 1, x + 1] == 1:
                ret = RemoveSpur(mask, y, x, y - 1, x + 1, 0, max_depth)
            elif mask[y + 1, x + 1] == 1:
                ret = RemoveSpur(mask, y, x, y + 1, x + 1, 0, max_depth)
            elif mask[y + 1, x - 1] == 1:
                ret = RemoveSpur(mask, y, x, y + 1, x - 1, 0, max_depth)
            elif mask[y - 1, x - 1] == 1:
                ret = RemoveSpur(mask, y, x, y - 1, x - 1, 0, max_depth)

            if ret == 1: mask[y, x] = 0

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def BranchLength(np.ndarray[dtype=np.uint8_t, ndim=2,
                            negative_indices=False, mode='c'] mask,
                 int y, int x, int y_prev, int x_prev):
    cdef:
        int ret, cn, vals[8], i

    # count neighbours
    vals[0] = mask[y - 1, x - 1]
    vals[1] = mask[y - 1, x]
    vals[2] = mask[y - 1, x + 1]
    vals[3] = mask[y, x + 1]
    vals[4] = mask[y + 1, x + 1]
    vals[5] = mask[y + 1, x]
    vals[6] = mask[y + 1, x - 1]
    vals[7] = mask[y, x - 1]
    cn = 0
    for i in range(8):
        cn += abs(vals[i] - vals[(i + 1) % 8])
    cn = cn // 2

    if cn != 1:
        #print 'end', y, x
        mask[y_prev, x_prev] = 1
        return 1

    mask[y, x] = 0

    if mask[y, x - 1] == 1:
        return BranchLength(mask, y, x - 1, y, x) + 1
    elif mask[y - 1, x] == 1:
        return BranchLength(mask, y - 1, x, y, x) + 1
    elif mask[y, x + 1] == 1:
        return BranchLength(mask, y, x + 1, y, x) + 1
    elif mask[y + 1, x] == 1:
        return BranchLength(mask, y + 1, x, y, x) + 1
    elif mask[y - 1, x + 1] == 1:
        return BranchLength(mask, y - 1, x + 1, y, x) + 1
    elif mask[y + 1, x + 1] == 1:
        return BranchLength(mask, y + 1, x + 1, y, x) + 1
    elif mask[y + 1, x - 1] == 1:
        return BranchLength(mask, y + 1, x - 1, y, x) + 1
    elif mask[y - 1, x - 1] == 1:
        return BranchLength(mask, y - 1, x - 1, y, x) + 1
    else:
        print 'huh?', y, x, cn
        return 0

@cython.boundscheck(False)
@cython.wraparound(False)
def BranchLengths(np.ndarray[dtype=np.uint8_t, ndim=2,
                             negative_indices=False, mode='c'] mask,
                  np.ndarray[dtype=np.int32_t, ndim=1,
                             negative_indices=False, mode='c'] counts):
    cdef:
        int h, w, x, y, ret, cn, vals[8], i, branch_num

    h = mask.shape[0]
    w = mask.shape[1]
    branch_num = 1

    for y in range(1, h - 1, 1):
        for x in range(1, w - 1, 1):
            if mask[y, x] == 0:
                continue

            # count neighbours
            vals[0] = mask[y - 1, x - 1]
            vals[1] = mask[y - 1, x]
            vals[2] = mask[y - 1, x + 1]
            vals[3] = mask[y, x + 1]
            vals[4] = mask[y + 1, x + 1]
            vals[5] = mask[y + 1, x]
            vals[6] = mask[y + 1, x - 1]
            vals[7] = mask[y, x - 1]
            cn = 0
            for i in range(8):
                cn += abs(vals[i] - vals[(i + 1) % 8])
            cn = cn // 2

            # only consider end points
            if cn != 1: continue

            #print 'start', y, x
            counts[branch_num] = BranchLength(mask, y, x, y, x)
            #print 'count', counts[branch_num]

            branch_num += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CountCrossings(np.ndarray[dtype=np.uint8_t, ndim=2,
                              negative_indices=False, mode='c'] mask,
                   np.ndarray[dtype=np.int32_t, ndim=1,
                              negative_indices=False, mode='c'] counts):
    cdef:
        int h, w, x, y, cn, vals[8], i

    h = mask.shape[0]
    w = mask.shape[1]
    num_endpoints = 0

    for y in range(1, h - 1, 1):
        for x in range(1, w - 1, 1):
            if mask[y, x] == 0: continue

            # compute crossing number
            vals[0] = mask[y - 1, x - 1]
            vals[1] = mask[y - 1, x]
            vals[2] = mask[y - 1, x + 1]
            vals[3] = mask[y, x + 1]
            vals[4] = mask[y + 1, x + 1]
            vals[5] = mask[y + 1, x]
            vals[6] = mask[y + 1, x - 1]
            vals[7] = mask[y, x - 1]
            cn = 0
            for i in range(8):
                cn += abs(vals[i] - vals[(i + 1) % 8])
            cn = cn // 2
            counts[cn] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def ComputeCrossings(np.ndarray[dtype=np.uint8_t, ndim=2,
                                negative_indices=False, mode='c'] mask,
                     np.ndarray[dtype=np.uint8_t, ndim=2,
                                negative_indices=False, mode='c'] counts):
    cdef:
        int h, w, x, y, cn, vals[8], i

    h = mask.shape[0]
    w = mask.shape[1]
    num_endpoints = 0

    for y in range(1, h - 1, 1):
        for x in range(1, w - 1, 1):
            if mask[y, x] == 0: continue

            # compute crossing number
            vals[0] = mask[y - 1, x - 1]
            vals[1] = mask[y - 1, x]
            vals[2] = mask[y - 1, x + 1]
            vals[3] = mask[y, x + 1]
            vals[4] = mask[y + 1, x + 1]
            vals[5] = mask[y + 1, x]
            vals[6] = mask[y + 1, x - 1]
            vals[7] = mask[y, x - 1]
            cn = 0
            for i in range(8):
                cn += abs(vals[i] - vals[(i + 1) % 8])
            cn = cn // 2
            counts[y, x] = cn

    return

def circle_perimeter(Py_ssize_t cy, Py_ssize_t cx, Py_ssize_t radius,
                     method='bresenham'):
    cdef list rr = list()
    cdef list cc = list()

    cdef Py_ssize_t x = 0
    cdef Py_ssize_t y = radius
    cdef Py_ssize_t d = 0
    cdef char cmethod
    if method == 'bresenham':
        d = 3 - 2 * radius
        cmethod = 'b'
    elif method == 'andres':
        d = radius - 1
        cmethod = 'a'
    else:
        raise ValueError('Wrong method')

    while y >= x:
        rr.extend([y, -y, y, -y, x, -x, x, -x])
        cc.extend([x, x, -x, -x, y, y, -y, -y])

        if cmethod == 'b':
            if d < 0:
                d += 4 * x + 6
            else:
                d += 4 * (x - y) + 10
                y -= 1
            x += 1
        elif cmethod == 'a':
            if d >= 2 * (x - 1):
                d = d - 2 * x
                x = x + 1
            elif d <= 2 * (radius - y):
                d = d + 2 * y - 1
                y = y - 1
            else:
                d = d + 2 * (y - x - 1)
                y = y - 1
                x = x + 1

    return np.array(rr, dtype=np.intp) + cy, np.array(cc, dtype=np.intp) + cx

def line_vote(np.ndarray[dtype=np.float32_t, ndim=2,
                         negative_indices=False, mode='c'] points,
              np.ndarray[dtype=np.int32_t, ndim=2,
                         negative_indices=False, mode='c'] acc,
              int q,
              float w,
              int min_dist):
    cdef:
        int num_points, s1, s2, l, r
        float x1, x2, y1, y2, slope

    #samples = points[::2, :]
    num_points = points.shape[0]
    for s1 in range(num_points):
        for s2 in range(s1 + 1, num_points):
            x1 = points[s1, 0]
            x2 = points[s2, 0]

            if x2 - x1 < min_dist: continue

            y1 = points[s1, 1]
            y2 = points[s2, 1]

            slope = (y2 - y1) / (x2 - x1)
            l = <int> round(y1 - (x1 * slope))
            r = <int> round(l + (w * slope))

            if q > l > -q and q > r > -q:
                acc[l + q, r + q] += 1

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def CircularBlobDetector(np.ndarray[dtype=np.float32_t, ndim=2,
                                    negative_indices=False, mode='c'] im,
                         np.ndarray[dtype=np.float32_t, ndim=1,
                                    negative_indices=False, mode='c'] mins,
                         np.ndarray[dtype=np.float32_t, ndim=1,
                                    negative_indices=False, mode='c'] maxs,
                         np.ndarray[dtype=np.int32_t, ndim=2,
                                    negative_indices=False, mode='c'] circles,
                         np.ndarray[dtype=np.float32_t, ndim=2,
                                    negative_indices=False, mode='c'] strength,
                         np.ndarray[dtype=np.uint8_t, ndim=2,
                                    negative_indices=False, mode='c'] radius,
                         float thresh):
    cdef:
        int h, w, x, y, r, R, D, D2, i, j
        float result, val
        float inner_val, diff
        #float result, prev_val

    h, w = im.shape[0], im.shape[1]
    D = circles.shape[0]
    D2 = (D - 1) // 2
    R = mins.shape[0]

    for y in range(R, h - R - 1):
        for x in range(R, w - R - 1):
            # skip if not local min
            if not (im[y, x] < im[y - 1, x] and im[y, x] < im[y + 1, x] and
                            im[y, x] < im[y, x - 1] and im[y, x] < im[y, x + 1]): continue

            # init mins & maxs
            for r in range(R):
                mins[r] = 1000
                maxs[r] = -1

            # get intensities at range of radius values
            for i in range(D):
                for j in range(D):
                    if circles[i, j] < 0: continue
                    val = im[y - D2 + i, x - D2 + j]
                    r = circles[i, j]
                    if val > maxs[r]: maxs[r] = val
                    if val < mins[r]: mins[r] = val

            result = 0
            for r in range(R):
                if r == 0:
                    inner_val = mins[0]
                elif maxs[r] > 0:
                    diff = mins[r] - inner_val
                    if diff > thresh and diff > strength[y, x]:
                        #result += mins[r] - inner_val
                        radius[y, x] = r
                        strength[y, x] = diff
                    inner_val = maxs[r]

                    #result = 0
                    #prev_val = -1
                    #for r in range(R):
                    #    # find median
                    #    if counts[r] == 0:
                    #        continue
                    #   elif r == 0:
                    #       prev_val = vals[0, 0]
                    #   else:
                    #       if False:
                    #           qsort(&vals[r, 0], counts[r], sizeof(np.float32_t), cmp)
                    #           result += vals[r, counts[r]//2] - prev_val
                    #           prev_val = vals[r, counts[r]//2]
                    #       else:
                    #           # min of outer circle - max of inner
                    #           pass

                    #output[y, x] = result

    return

@cython.boundscheck(False)
@cython.wraparound(False)
def cal_img_grad(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] delta,
                 np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] gS,
                 float kappa_sqr):
    cdef:
        int w, h, k, y, x

    ## calculate: gS = kappa_sqr/(kappa_sqr + deltaS**2.).
    h = delta.shape[0]
    w = delta.shape[1]

    # iterate through rows of delta
    for y in range(h):
        # iterate through columns of delta
        for x in range(w):
            gS[y, x] = kappa_sqr / (kappa_sqr + (delta[y, x] * delta[y, x]))
    return

@cython.boundscheck(False)
@cython.wraparound(False)
def update_img_grad(np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] img,
                    np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] NS,
                    np.ndarray[dtype=np.float32_t, ndim=2, negative_indices=False, mode='c'] EW,
                    float gamma):
    cdef:
        int w, h, y, x

    h = img.shape[0]
    w = img.shape[1]
    for y in range(h):
        for x in range(w):
            img[y, x] += gamma * (NS[y, x] + EW[y, x])

    return
