r"""


C:\Users\Neil\Miniconda2\envs\BT\python.exe compile_cc_label.py build_ext --inplace  --compiler=msvc
"""
import numpy as np
cimport numpy as np
cimport cython

#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef inline merge2(int u, int v,
#                   np.ndarray[dtype=np.int32_t, ndim=1,
#                              negative_indices=False, mode='c'] aRTable,
#                   np.ndarray[dtype=np.int32_t, ndim=1,
#                              negative_indices=False, mode='c'] aNext,
#                   np.ndarray[dtype=np.int32_t, ndim=1,
#                              negative_indices=False, mode='c'] aTail):
#
#    cdef:
#        int i
#
#    if (u<v):
#        i = v
#        while (i>-1):
#            aRTable[i] = u
#            i = aNext[i]
#        aNext[aTail[u]] = v
#        aTail[u] = aTail[v]
#    elif (u>v):
#        i = u
#        while (i>-1):
#            aRTable[i] = v
#            i = aNext[i]
#        aNext[aTail[v]] = u
#        aTail[v] = aTail[u]
#
#
#@cython.boundscheck(False)
#@cython.wraparound(False)
#cdef inline merge3(int u, int v, int k,
#                   np.ndarray[dtype=np.int32_t, ndim=1,
#                              negative_indices=False, mode='c'] aRTable,
#                   np.ndarray[dtype=np.int32_t, ndim=1,
#                              negative_indices=False, mode='c'] aNext,
#                   np.ndarray[dtype=np.int32_t, ndim=1,
#                              negative_indices=False, mode='c'] aTail):
#
#    cdef:
#        int i
#
#    if (u<v):
#        i = v
#        while (i>-1):
#            aRTable[i] = u
#            i = aNext[i]
#        aNext[aTail[u]] = v
#        aTail[u] = aTail[v]
#        k = aRTable[k]
#        if (u<k):
#            i = k
#            while (i>-1):
#                aRTable[i] = u
#                i = aNext[i]
#            aNext[aTail[u]] = k
#            aTail[u] = aTail[k]
#        elif (u>k):
#            i = u
#            while (i>-1):
#                aRTable[i] = k
#                i = aNext[i]
#            aNext[aTail[k]] = u
#            aTail[k] = aTail[u]
#    elif (u>v):
#        i = u
#        while (i>-1):
#            aRTable[i] = v
#            i = aNext[i]
#        aNext[aTail[v]] = u
#        aTail[v] = aTail[u]
#        k = aRTable[k]
#        if (v<k):
#            i = k
#            while (i>-1):
#                aRTable[i] = v
#                i = aNext[i]
#            aNext[aTail[v]] = k
#            aTail[v] = aTail[k]
#        elif (v>k):
#            i = v
#            while (i>-1):
#                aRTable[i] = k
#                i = aNext[i]
#            aNext[aTail[k]] = v
#            aTail[k] = aTail[v]
#    else:
#        k = aRTable[k]
#        if (u<k):
#            i = k
#            while (i>-1):
#                aRTable[i] = u
#                i = aNext[i]
#            aNext[aTail[u]] = k
#            aTail[u] = aTail[k]
#        elif (u>k):
#            i = u
#            while (i>-1):
#                aRTable[i] = k
#                i = aNext[i]
#            aNext[aTail[k]] = u
#            aTail[k] = aTail[u]


@cython.boundscheck(False)
@cython.wraparound(False)
def CCLabel(np.ndarray[dtype=np.uint8_t, ndim=2,
                       negative_indices=False, mode='c'] img,
            np.ndarray[dtype=np.int32_t, ndim=2,
                       negative_indices=False, mode='c'] results,
            np.ndarray[dtype=np.int32_t, ndim=1,
                       negative_indices=False, mode='c'] aRTable,
            np.ndarray[dtype=np.int32_t, ndim=1,
                       negative_indices=False, mode='c'] aNext,
            np.ndarray[dtype=np.int32_t, ndim=1,
                       negative_indices=False, mode='c'] aTail):

    cdef:
        int w, h, y, x
        int lx, u, v, k
        int new_label, cur_label, num_labels, label
        int i

    new_label = 0
    h = img.shape[0]
    w = img.shape[1]
    for y in range(2, h-2, 2):
        for x in range(2, w-2, 2):
            if img[y, x] == 1:
                if img[y, x-1] == 1:
                    if img[y-1, x+2] == 1:
                        if img[y-1, x+1] == 1:
                            if img[y-1, x] == 1:
                                lx = results[y-2, x]
                            else:
                                if img[y-2, x] == 1:
                                    if img[y-1, x-1] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-1, x-2] == 1:
                                            if img[y-2, x-1] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                else:
                                    lx = results[y, x-2]
                                    u = aRTable[lx]
                                    v = aRTable[results[y-2, x+2]]
                                    if (u<v):
                                        i = v
                                        while (i>-1):
                                            aRTable[i] = u
                                            i = aNext[i]
                                        aNext[aTail[u]] = v
                                        aTail[u] = aTail[v]
                                    elif (u>v):
                                        i = u
                                        while (i>-1):
                                            aRTable[i] = v
                                            i = aNext[i]
                                        aNext[aTail[v]] = u
                                        aTail[v] = aTail[u]
                        else:
                            if img[y, x+1] == 1:
                                if img[y-2, x+1] == 1:
                                    if img[y-1, x] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-2, x] == 1:
                                            if img[y-1, x-1] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-1, x-2] == 1:
                                                    if img[y-2, x-1] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                else:
                                    lx = results[y, x-2]
                                    u = aRTable[lx]
                                    v = aRTable[results[y-2, x+2]]
                                    if (u<v):
                                        i = v
                                        while (i>-1):
                                            aRTable[i] = u
                                            i = aNext[i]
                                        aNext[aTail[u]] = v
                                        aTail[u] = aTail[v]
                                    elif (u>v):
                                        i = u
                                        while (i>-1):
                                            aRTable[i] = v
                                            i = aNext[i]
                                        aNext[aTail[v]] = u
                                        aTail[v] = aTail[u]
                            else:
                                if img[y-1, x] == 1:
                                    lx = results[y-2, x]
                                else:
                                    if img[y-2, x] == 1:
                                        if img[y-1, x-1] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-1, x-2] == 1:
                                                if img[y-2, x-1] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    lx = results[y, x-2]
                                            else:
                                                lx = results[y, x-2]
                                    else:
                                        lx = results[y, x-2]
                    else:
                        if img[y-1, x] == 1:
                            lx = results[y-2, x]
                        else:
                            if img[y-2, x] == 1:
                                if img[y-1, x-1] == 1:
                                    lx = results[y-2, x]
                                else:
                                    if img[y-1, x-2] == 1:
                                        if img[y-2, x-1] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-1, x+1] == 1:
                                                if img[y-2, x+2] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                    else:
                                        if img[y-1, x+1] == 1:
                                            if img[y-2, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                            else:
                                if img[y-1, x+1] == 1:
                                    if img[y-2, x+2] == 1:
                                        lx = results[y, x-2]
                                        u = aRTable[lx]
                                        v = aRTable[results[y-2, x+2]]
                                        if (u<v):
                                            i = v
                                            while (i>-1):
                                                aRTable[i] = u
                                                i = aNext[i]
                                            aNext[aTail[u]] = v
                                            aTail[u] = aTail[v]
                                        elif (u>v):
                                            i = u
                                            while (i>-1):
                                                aRTable[i] = v
                                                i = aNext[i]
                                            aNext[aTail[v]] = u
                                            aTail[v] = aTail[u]
                                    else:
                                        lx = results[y, x-2]
                                        u = aRTable[lx]
                                        v = aRTable[results[y-2, x]]
                                        if (u<v):
                                            i = v
                                            while (i>-1):
                                                aRTable[i] = u
                                                i = aNext[i]
                                            aNext[aTail[u]] = v
                                            aTail[u] = aTail[v]
                                        elif (u>v):
                                            i = u
                                            while (i>-1):
                                                aRTable[i] = v
                                                i = aNext[i]
                                            aNext[aTail[v]] = u
                                            aTail[v] = aTail[u]
                                else:
                                    lx = results[y, x-2]
                else:
                    if img[y-1, x+1] == 1:
                        if img[y-1, x-1] == 1:
                            if img[y, x-2] == 1:
                                if img[y-1, x] == 1:
                                    lx = results[y-2, x]
                                else:
                                    if img[y-2, x] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-1, x+2] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                            else:
                                if img[y+1, x-1] == 1:
                                    if img[y-1, x+2] == 1:
                                        if img[y-1, x] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x-2]]
                                                k = results[y-2, x+2]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                    k = aRTable[k]
                                                    if (u<k):
                                                        i = k
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = k
                                                        aTail[u] = aTail[k]
                                                    elif (u>k):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = k
                                                            i = aNext[i]
                                                        aNext[aTail[k]] = u
                                                        aTail[k] = aTail[u]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                                    k = aRTable[k]
                                                    if (v<k):
                                                        i = k
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = k
                                                        aTail[v] = aTail[k]
                                                    elif (v>k):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = k
                                                            i = aNext[i]
                                                        aNext[aTail[k]] = v
                                                        aTail[k] = aTail[v]
                                                else:
                                                    k = aRTable[k]
                                                    if (u<k):
                                                        i = k
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = k
                                                        aTail[u] = aTail[k]
                                                    elif (u>k):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = k
                                                            i = aNext[i]
                                                        aNext[aTail[k]] = u
                                                        aTail[k] = aTail[u]
                                    else:
                                        if img[y-1, x] == 1:
                                            if img[y-2, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x+2] == 1:
                                                if img[y-2, x] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x-2]]
                                                    k = results[y-2, x+2]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                        k = aRTable[k]
                                                        if (u<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = k
                                                            aTail[u] = aTail[k]
                                                        elif (u>k):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = u
                                                            aTail[k] = aTail[u]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                        k = aRTable[k]
                                                        if (v<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = k
                                                            aTail[v] = aTail[k]
                                                        elif (v>k):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = v
                                                            aTail[k] = aTail[v]
                                                    else:
                                                        k = aRTable[k]
                                                        if (u<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = k
                                                            aTail[u] = aTail[k]
                                                        elif (u>k):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = u
                                                            aTail[k] = aTail[u]
                                            else:
                                                if img[y-2, x] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x-2]]
                                                    k = results[y-2, x]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                        k = aRTable[k]
                                                        if (u<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = k
                                                            aTail[u] = aTail[k]
                                                        elif (u>k):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = u
                                                            aTail[k] = aTail[u]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                        k = aRTable[k]
                                                        if (v<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = k
                                                            aTail[v] = aTail[k]
                                                        elif (v>k):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = v
                                                            aTail[k] = aTail[v]
                                                    else:
                                                        k = aRTable[k]
                                                        if (u<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = k
                                                            aTail[u] = aTail[k]
                                                        elif (u>k):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = u
                                                            aTail[k] = aTail[u]
                                else:
                                    if img[y-1, x] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-2, x] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-1, x+2] == 1:
                                                lx = results[y-2, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                if img[y-2, x+2] == 1:
                                                    lx = results[y-2, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y-2, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                        else:
                            if img[y+1, x-1] == 1:
                                if img[y, x-2] == 1:
                                    if img[y-1, x-2] == 1:
                                        if img[y-2, x-1] == 1:
                                            if img[y-1, x] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-2, x] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-1, x+2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        if img[y-2, x+2] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-1, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                if img[y-2, x+2] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                    else:
                                        if img[y-1, x+2] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                else:
                                    if img[y-1, x+2] == 1:
                                        lx = results[y, x-2]
                                        u = aRTable[lx]
                                        v = aRTable[results[y-2, x+2]]
                                        if (u<v):
                                            i = v
                                            while (i>-1):
                                                aRTable[i] = u
                                                i = aNext[i]
                                            aNext[aTail[u]] = v
                                            aTail[u] = aTail[v]
                                        elif (u>v):
                                            i = u
                                            while (i>-1):
                                                aRTable[i] = v
                                                i = aNext[i]
                                            aNext[aTail[v]] = u
                                            aTail[v] = aTail[u]
                                    else:
                                        if img[y-2, x+2] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                            else:
                                lx = results[y-2, x]
                    else:
                        if img[y-1, x+2] == 1:
                            if img[y, x+1] == 1:
                                if img[y-1, x-1] == 1:
                                    if img[y, x-2] == 1:
                                        if img[y-2, x+1] == 1:
                                            if img[y-1, x] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-2, x] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                    else:
                                        if img[y+1, x-1] == 1:
                                            if img[y-2, x+1] == 1:
                                                if img[y-1, x] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    if img[y-2, x] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x-2]]
                                                        k = results[y-2, x+2]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                            k = aRTable[k]
                                                            if (v<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = k
                                                                aTail[v] = aTail[k]
                                                            elif (v>k):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = v
                                                                aTail[k] = aTail[v]
                                                        else:
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x-2]]
                                                k = results[y-2, x+2]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                    k = aRTable[k]
                                                    if (u<k):
                                                        i = k
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = k
                                                        aTail[u] = aTail[k]
                                                    elif (u>k):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = k
                                                            i = aNext[i]
                                                        aNext[aTail[k]] = u
                                                        aTail[k] = aTail[u]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                                    k = aRTable[k]
                                                    if (v<k):
                                                        i = k
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = k
                                                        aTail[v] = aTail[k]
                                                    elif (v>k):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = k
                                                            i = aNext[i]
                                                        aNext[aTail[k]] = v
                                                        aTail[k] = aTail[v]
                                                else:
                                                    k = aRTable[k]
                                                    if (u<k):
                                                        i = k
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = k
                                                        aTail[u] = aTail[k]
                                                    elif (u>k):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = k
                                                            i = aNext[i]
                                                        aNext[aTail[k]] = u
                                                        aTail[k] = aTail[u]
                                        else:
                                            if img[y-2, x+1] == 1:
                                                if img[y-1, x] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-2, x] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        lx = results[y-2, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                            else:
                                                lx = results[y-2, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                else:
                                    if img[y-2, x+1] == 1:
                                        if img[y+1, x-1] == 1:
                                            if img[y, x-2] == 1:
                                                if img[y-1, x-2] == 1:
                                                    if img[y-2, x-1] == 1:
                                                        if img[y-1, x] == 1:
                                                            lx = results[y-2, x]
                                                        else:
                                                            if img[y-2, x] == 1:
                                                                lx = results[y-2, x]
                                                            else:
                                                                lx = results[y, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x+2]]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            lx = results[y-2, x]
                                    else:
                                        if img[y-1, x] == 1:
                                            if img[y-2, x-1] == 1:
                                                if img[y, x-2] == 1:
                                                    if img[y-1, x-2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        if img[y+1, x-1] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x-2]]
                                                            k = results[y-2, x+2]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                                k = aRTable[k]
                                                                if (u<k):
                                                                    i = k
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = k
                                                                    aTail[u] = aTail[k]
                                                                elif (u>k):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = k
                                                                        i = aNext[i]
                                                                    aNext[aTail[k]] = u
                                                                    aTail[k] = aTail[u]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                                k = aRTable[k]
                                                                if (v<k):
                                                                    i = k
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = k
                                                                    aTail[v] = aTail[k]
                                                                elif (v>k):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = k
                                                                        i = aNext[i]
                                                                    aNext[aTail[k]] = v
                                                                    aTail[k] = aTail[v]
                                                            else:
                                                                k = aRTable[k]
                                                                if (u<k):
                                                                    i = k
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = k
                                                                    aTail[u] = aTail[k]
                                                                elif (u>k):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = k
                                                                        i = aNext[i]
                                                                    aNext[aTail[k]] = u
                                                                    aTail[k] = aTail[u]
                                                        else:
                                                            lx = results[y-2, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                else:
                                                    if img[y+1, x-1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x-2]]
                                                        k = results[y-2, x+2]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                            k = aRTable[k]
                                                            if (v<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = k
                                                                aTail[v] = aTail[k]
                                                            elif (v>k):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = v
                                                                aTail[k] = aTail[v]
                                                        else:
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                    else:
                                                        lx = results[y-2, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                            else:
                                                if img[y+1, x-1] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    k = results[y-2, x+2]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                        k = aRTable[k]
                                                        if (u<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = k
                                                            aTail[u] = aTail[k]
                                                        elif (u>k):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = u
                                                            aTail[k] = aTail[u]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                        k = aRTable[k]
                                                        if (v<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = k
                                                            aTail[v] = aTail[k]
                                                        elif (v>k):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = v
                                                            aTail[k] = aTail[v]
                                                    else:
                                                        k = aRTable[k]
                                                        if (u<k):
                                                            i = k
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = k
                                                            aTail[u] = aTail[k]
                                                        elif (u>k):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = k
                                                                i = aNext[i]
                                                            aNext[aTail[k]] = u
                                                            aTail[k] = aTail[u]
                                                else:
                                                    lx = results[y-2, x]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                        else:
                                            if img[y+1, x-1] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y-2, x+2]
                            else:
                                if img[y+1, x-1] == 1:
                                    if img[y, x-2] == 1:
                                        if img[y-1, x] == 1:
                                            if img[y-1, x-1] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-1, x-2] == 1:
                                                    if img[y-2, x-1] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        if img[y-2, x+1] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                else:
                                                    if img[y-2, x+1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x] == 1:
                                                if img[y-1, x-1] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-1, x-2] == 1:
                                                        if img[y-2, x-1] == 1:
                                                            lx = results[y-2, x]
                                                        else:
                                                            lx = results[y, x-2]
                                                    else:
                                                        lx = results[y, x-2]
                                            else:
                                                lx = results[y, x-2]
                                    else:
                                        if img[y-1, x] == 1:
                                            if img[y-2, x+1] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            if img[y-1, x-1] == 1:
                                                if img[y-2, x] == 1:
                                                    if img[y-2, x+1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx];
                                                    v = aRTable[results[y-2, x-2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                else:
                                    if img[y-1, x] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-1, x-1] == 1:
                                            if img[y-2, x] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y, x-2] == 1:
                                                    lx = results[y, x-2]
                                                else:
                                                    lx = results[y-2, x-2]
                                        else:
                                            new_label += 1
                                            lx = new_label
                                            aRTable[lx] = lx
                                            aNext[lx] = -1
                                            aTail[lx] = lx
                        else:
                            if img[y+1, x-1] == 1:
                                if img[y, x-2] == 1:
                                    if img[y-1, x] == 1:
                                        if img[y-1, x-1] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-1, x-2] == 1:
                                                if img[y-2, x-1] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-2, x+2] == 1:
                                                        if img[y-2, x+1] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                            else:
                                                if img[y-2, x+2] == 1:
                                                    if img[y-2, x+1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                    else:
                                        if img[y-2, x] == 1:
                                            if img[y-1, x-1] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-1, x-2] == 1:
                                                    if img[y-2, x-1] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        lx = results[y, x-2]
                                                else:
                                                    lx = results[y, x-2]
                                        else:
                                            lx = results[y, x-2]
                                else:
                                    if img[y-1, x] == 1:
                                        if img[y-2, x+2] == 1:
                                            if img[y-2, x+1] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                    else:
                                        if img[y-1, x-1] == 1:
                                            if img[y-2, x] == 1:
                                                if img[y-2, x+2] == 1:
                                                    if img[y-2, x+1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx];
                                                v = aRTable[results[y-2, x-2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            lx = results[y, x-2]
                            else:
                                if img[y-1, x] == 1:
                                    lx = results[y-2, x]
                                else:
                                    if img[y-1, x-1] == 1:
                                        if img[y-2, x] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y, x-2] == 1:
                                                lx = results[y, x-2]
                                            else:
                                                lx = results[y-2, x-2]
                                    else:
                                        new_label += 1
                                        lx = new_label
                                        aRTable[lx] = lx
                                        aNext[lx] = -1
                                        aTail[lx] = lx
            else:
                if img[y, x+1] == 1:
                    if img[y-1, x+1] == 1:
                        if img[y+1, x] == 1:
                            if img[y, x-1]==1:
                                if img[y-1, x] == 1:
                                    lx = results[y-2, x]
                                else:
                                    if img[y-2, x] == 1:
                                        if img[y-1, x-1] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-1, x-2] == 1:
                                                if img[y-2, x-1] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-1, x+2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        if img[y-2, x+2] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                            else:
                                                if img[y-1, x+2] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    if img[y-2, x+2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                    else:
                                        if img[y-1, x+2] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                            else:
                                if img[y+1, x-1] == 1:
                                    if img[y, x-2] == 1:
                                        if img[y-1, x-1] == 1:
                                            if img[y-1, x] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-2, x] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-1, x+2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        if img[y-2, x+2] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-1, x-2] == 1:
                                                if img[y-2, x-1] == 1:
                                                    if img[y-1, x] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        if img[y-2, x] == 1:
                                                            lx = results[y-2, x]
                                                        else:
                                                            if img[y-1, x+2] == 1:
                                                                lx = results[y, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x+2]]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                            else:
                                                                if img[y-2, x+2] == 1:
                                                                    lx = results[y, x-2]
                                                                    u = aRTable[lx]
                                                                    v = aRTable[results[y-2, x+2]]
                                                                    if (u<v):
                                                                        i = v
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = v
                                                                        aTail[u] = aTail[v]
                                                                    elif (u>v):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = v
                                                                            i = aNext[i]
                                                                        aNext[aTail[v]] = u
                                                                        aTail[v] = aTail[u]
                                                                else:
                                                                    lx = results[y, x-2]
                                                                    u = aRTable[lx]
                                                                    v = aRTable[results[y-2, x]]
                                                                    if (u<v):
                                                                        i = v
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = v
                                                                        aTail[u] = aTail[v]
                                                                    elif (u>v):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = v
                                                                            i = aNext[i]
                                                                        aNext[aTail[v]] = u
                                                                        aTail[v] = aTail[u]
                                                else:
                                                    if img[y-1, x+2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        if img[y-2, x+2] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                            else:
                                                if img[y-1, x+2] == 1:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                                else:
                                                    if img[y-2, x+2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                    else:
                                        if img[y-1, x+2] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x+2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                else:
                                    lx = results[y-2, x]
                        else:
                            lx = results[y-2, x]
                    else:
                        if img[y-1, x+2] == 1:
                            if img[y-2, x+1] == 1:
                                if img[y+1, x] == 1:
                                    if img[y, x-1]==1:
                                        if img[y-1, x] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-2, x] == 1:
                                                if img[y-1, x-1] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-1, x-2] == 1:
                                                        if img[y-2, x-1] == 1:
                                                            lx = results[y-2, x]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                    else:
                                        if img[y+1, x-1] == 1:
                                            if img[y, x-2] == 1:
                                                if img[y-1, x-1] == 1:
                                                    if img[y-1, x] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        if img[y-2, x] == 1:
                                                            lx = results[y-2, x]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                else:
                                                    if img[y-1, x-2] == 1:
                                                        if img[y-2, x-1] == 1:
                                                            if img[y-1, x] == 1:
                                                                lx = results[y-2, x]
                                                            else:
                                                                if img[y-2, x] == 1:
                                                                    lx = results[y-2, x]
                                                                else:
                                                                    lx = results[y, x-2]
                                                                    u = aRTable[lx]
                                                                    v = aRTable[results[y-2, x+2]]
                                                                    if (u<v):
                                                                        i = v
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = v
                                                                        aTail[u] = aTail[v]
                                                                    elif (u>v):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = v
                                                                            i = aNext[i]
                                                                        aNext[aTail[v]] = u
                                                                        aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                        else:
                                            lx = results[y-2, x]
                                else:
                                    lx = results[y-2, x]
                            else:
                                if img[y-1, x] == 1:
                                    if img[y, x-1]==1:
                                        lx = results[y, x-2]
                                        u = aRTable[lx]
                                        v = aRTable[results[y-2, x+2]]
                                        if (u<v):
                                            i = v
                                            while (i>-1):
                                                aRTable[i] = u
                                                i = aNext[i]
                                            aNext[aTail[u]] = v
                                            aTail[u] = aTail[v]
                                        elif (u>v):
                                            i = u
                                            while (i>-1):
                                                aRTable[i] = v
                                                i = aNext[i]
                                            aNext[aTail[v]] = u
                                            aTail[v] = aTail[u]
                                    else:
                                        if img[y-1, x-1] == 1:
                                            if img[y, x-2] == 1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                if img[y+1, x] == 1:
                                                    if img[y+1, x-1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x-2]]
                                                        k = results[y-2, x+2]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                            k = aRTable[k]
                                                            if (v<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = k
                                                                aTail[v] = aTail[k]
                                                            elif (v>k):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = v
                                                                aTail[k] = aTail[v]
                                                        else:
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                    else:
                                                        lx = results[y-2, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y-2, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                        else:
                                            if img[y-2, x-1] == 1:
                                                if img[y, x-2] == 1:
                                                    if img[y-1, x-2] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        if img[y+1, x] == 1:
                                                            if img[y+1, x-1] == 1:
                                                                lx = results[y, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x-2]]
                                                                k = results[y-2, x+2]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                    k = aRTable[k]
                                                                    if (u<k):
                                                                        i = k
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = k
                                                                        aTail[u] = aTail[k]
                                                                    elif (u>k):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = k
                                                                            i = aNext[i]
                                                                        aNext[aTail[k]] = u
                                                                        aTail[k] = aTail[u]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                                    k = aRTable[k]
                                                                    if (v<k):
                                                                        i = k
                                                                        while (i>-1):
                                                                            aRTable[i] = v
                                                                            i = aNext[i]
                                                                        aNext[aTail[v]] = k
                                                                        aTail[v] = aTail[k]
                                                                    elif (v>k):
                                                                        i = v
                                                                        while (i>-1):
                                                                            aRTable[i] = k
                                                                            i = aNext[i]
                                                                        aNext[aTail[k]] = v
                                                                        aTail[k] = aTail[v]
                                                                else:
                                                                    k = aRTable[k]
                                                                    if (u<k):
                                                                        i = k
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = k
                                                                        aTail[u] = aTail[k]
                                                                    elif (u>k):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = k
                                                                            i = aNext[i]
                                                                        aNext[aTail[k]] = u
                                                                        aTail[k] = aTail[u]
                                                            else:
                                                                lx = results[y-2, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x+2]]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y-2, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                else:
                                                    if img[y+1, x] == 1:
                                                        if img[y+1, x-1] == 1:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x-2]]
                                                            k = results[y-2, x+2]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                                k = aRTable[k]
                                                                if (u<k):
                                                                    i = k
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = k
                                                                    aTail[u] = aTail[k]
                                                                elif (u>k):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = k
                                                                        i = aNext[i]
                                                                    aNext[aTail[k]] = u
                                                                    aTail[k] = aTail[u]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                                k = aRTable[k]
                                                                if (v<k):
                                                                    i = k
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = k
                                                                    aTail[v] = aTail[k]
                                                                elif (v>k):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = k
                                                                        i = aNext[i]
                                                                    aNext[aTail[k]] = v
                                                                    aTail[k] = aTail[v]
                                                            else:
                                                                k = aRTable[k]
                                                                if (u<k):
                                                                    i = k
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = k
                                                                    aTail[u] = aTail[k]
                                                                elif (u>k):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = k
                                                                        i = aNext[i]
                                                                    aNext[aTail[k]] = u
                                                                    aTail[k] = aTail[u]
                                                        else:
                                                            lx = results[y-2, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x+2]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y-2, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                            else:
                                                if img[y+1, x] == 1:
                                                    if img[y+1, x-1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        k = results[y-2, x+2]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                            k = aRTable[k]
                                                            if (v<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = k
                                                                aTail[v] = aTail[k]
                                                            elif (v>k):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = v
                                                                aTail[k] = aTail[v]
                                                        else:
                                                            k = aRTable[k]
                                                            if (u<k):
                                                                i = k
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = k
                                                                aTail[u] = aTail[k]
                                                            elif (u>k):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = k
                                                                    i = aNext[i]
                                                                aNext[aTail[k]] = u
                                                                aTail[k] = aTail[u]
                                                    else:
                                                        lx = results[y-2, x]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y-2, x]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x+2]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                else:
                                    if img[y+1, x] == 1:
                                        if img[y+1, x-1] == 1:
                                            lx = results[y, x-2]
                                            u = aRTable[lx]
                                            v = aRTable[results[y-2, x+2]]
                                            if (u<v):
                                                i = v
                                                while (i>-1):
                                                    aRTable[i] = u
                                                    i = aNext[i]
                                                aNext[aTail[u]] = v
                                                aTail[u] = aTail[v]
                                            elif (u>v):
                                                i = u
                                                while (i>-1):
                                                    aRTable[i] = v
                                                    i = aNext[i]
                                                aNext[aTail[v]] = u
                                                aTail[v] = aTail[u]
                                        else:
                                            if img[y, x-1]==1:
                                                lx = results[y, x-2]
                                                u = aRTable[lx]
                                                v = aRTable[results[y-2, x+2]]
                                                if (u<v):
                                                    i = v
                                                    while (i>-1):
                                                        aRTable[i] = u
                                                        i = aNext[i]
                                                    aNext[aTail[u]] = v
                                                    aTail[u] = aTail[v]
                                                elif (u>v):
                                                    i = u
                                                    while (i>-1):
                                                        aRTable[i] = v
                                                        i = aNext[i]
                                                    aNext[aTail[v]] = u
                                                    aTail[v] = aTail[u]
                                            else:
                                                lx = results[y-2, x+2]
                                    else:
                                        lx = results[y-2, x+2]
                        else:
                            if img[y+1, x] == 1:
                                if img[y, x-1]==1:
                                    if img[y-1, x] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-2, x] == 1:
                                            if img[y-1, x-1] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                if img[y-1, x-2] == 1:
                                                    if img[y-2, x-1] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        lx = results[y, x-2]
                                                else:
                                                    lx = results[y, x-2]
                                        else:
                                            lx = results[y, x-2]
                                else:
                                    if img[y+1, x-1] == 1:
                                        if img[y, x-2] == 1:
                                            if img[y-1, x] == 1:
                                                if img[y-1, x-1] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-1, x-2] == 1:
                                                        if img[y-2, x-1] == 1:
                                                            lx = results[y-2, x]
                                                        else:
                                                            if img[y-2, x+2] == 1:
                                                                if img[y-2, x+1] == 1:
                                                                    lx = results[y, x-2]
                                                                    u = aRTable[lx]
                                                                    v = aRTable[results[y-2, x+2]]
                                                                    if (u<v):
                                                                        i = v
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = v
                                                                        aTail[u] = aTail[v]
                                                                    elif (u>v):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = v
                                                                            i = aNext[i]
                                                                        aNext[aTail[v]] = u
                                                                        aTail[v] = aTail[u]
                                                                else:
                                                                    lx = results[y, x-2]
                                                                    u = aRTable[lx]
                                                                    v = aRTable[results[y-2, x]]
                                                                    if (u<v):
                                                                        i = v
                                                                        while (i>-1):
                                                                            aRTable[i] = u
                                                                            i = aNext[i]
                                                                        aNext[aTail[u]] = v
                                                                        aTail[u] = aTail[v]
                                                                    elif (u>v):
                                                                        i = u
                                                                        while (i>-1):
                                                                            aRTable[i] = v
                                                                            i = aNext[i]
                                                                        aNext[aTail[v]] = u
                                                                        aTail[v] = aTail[u]
                                                            else:
                                                                lx = results[y, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x]]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                    else:
                                                        if img[y-2, x+2] == 1:
                                                            if img[y-2, x+1] == 1:
                                                                lx = results[y, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x+2]]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                            else:
                                                                lx = results[y, x-2]
                                                                u = aRTable[lx]
                                                                v = aRTable[results[y-2, x]]
                                                                if (u<v):
                                                                    i = v
                                                                    while (i>-1):
                                                                        aRTable[i] = u
                                                                        i = aNext[i]
                                                                    aNext[aTail[u]] = v
                                                                    aTail[u] = aTail[v]
                                                                elif (u>v):
                                                                    i = u
                                                                    while (i>-1):
                                                                        aRTable[i] = v
                                                                        i = aNext[i]
                                                                    aNext[aTail[v]] = u
                                                                    aTail[v] = aTail[u]
                                                        else:
                                                            lx = results[y, x-2]
                                                            u = aRTable[lx]
                                                            v = aRTable[results[y-2, x]]
                                                            if (u<v):
                                                                i = v
                                                                while (i>-1):
                                                                    aRTable[i] = u
                                                                    i = aNext[i]
                                                                aNext[aTail[u]] = v
                                                                aTail[u] = aTail[v]
                                                            elif (u>v):
                                                                i = u
                                                                while (i>-1):
                                                                    aRTable[i] = v
                                                                    i = aNext[i]
                                                                aNext[aTail[v]] = u
                                                                aTail[v] = aTail[u]
                                            else:
                                                if img[y-2, x] == 1:
                                                    if img[y-1, x-1] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        if img[y-1, x-2] == 1:
                                                            if img[y-2, x-1] == 1:
                                                                lx = results[y-2, x]
                                                            else:
                                                                lx = results[y, x-2]
                                                        else:
                                                            lx = results[y, x-2]
                                                else:
                                                    lx = results[y, x-2]
                                        else:
                                            if img[y-1, x] == 1:
                                                if img[y-2, x+2] == 1:
                                                    if img[y-2, x+1] == 1:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x+2]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                    else:
                                                        lx = results[y, x-2]
                                                        u = aRTable[lx]
                                                        v = aRTable[results[y-2, x]]
                                                        if (u<v):
                                                            i = v
                                                            while (i>-1):
                                                                aRTable[i] = u
                                                                i = aNext[i]
                                                            aNext[aTail[u]] = v
                                                            aTail[u] = aTail[v]
                                                        elif (u>v):
                                                            i = u
                                                            while (i>-1):
                                                                aRTable[i] = v
                                                                i = aNext[i]
                                                            aNext[aTail[v]] = u
                                                            aTail[v] = aTail[u]
                                                else:
                                                    lx = results[y, x-2]
                                                    u = aRTable[lx]
                                                    v = aRTable[results[y-2, x]]
                                                    if (u<v):
                                                        i = v
                                                        while (i>-1):
                                                            aRTable[i] = u
                                                            i = aNext[i]
                                                        aNext[aTail[u]] = v
                                                        aTail[u] = aTail[v]
                                                    elif (u>v):
                                                        i = u
                                                        while (i>-1):
                                                            aRTable[i] = v
                                                            i = aNext[i]
                                                        aNext[aTail[v]] = u
                                                        aTail[v] = aTail[u]
                                            else:
                                                lx = results[y, x-2]
                                    else:
                                        if img[y-1, x] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            new_label += 1
                                            lx = new_label
                                            aRTable[lx] = lx
                                            aNext[lx] = -1
                                            aTail[lx] = lx
                            else:
                                if img[y-1, x] == 1:
                                    lx = results[y-2, x]
                                else:
                                    new_label += 1
                                    lx = new_label
                                    aRTable[lx] = lx
                                    aNext[lx] = -1
                                    aTail[lx] = lx
                else:
                    if img[y+1, x] == 1:
                        if img[y, x-1]==1:
                            if img[y-1, x] == 1:
                                lx = results[y-2, x]
                            else:
                                if img[y-2, x] == 1:
                                    if img[y-1, x-1] == 1:
                                        lx = results[y-2, x]
                                    else:
                                        if img[y-1, x-2] == 1:
                                            if img[y-2, x-1] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                lx = results[y, x-2]
                                        else:
                                            lx = results[y, x-2]
                                else:
                                    lx = results[y, x-2]
                        else:
                            if img[y+1, x-1] == 1:
                                if img[y, x-2] == 1:
                                    if img[y-1, x-1] == 1:
                                        if img[y-1, x] == 1:
                                            lx = results[y-2, x]
                                        else:
                                            if img[y-2, x] == 1:
                                                lx = results[y-2, x]
                                            else:
                                                lx = results[y, x-2]
                                    else:
                                        if img[y-1, x-2] == 1:
                                            if img[y-2, x-1] == 1:
                                                if img[y-1, x] == 1:
                                                    lx = results[y-2, x]
                                                else:
                                                    if img[y-2, x] == 1:
                                                        lx = results[y-2, x]
                                                    else:
                                                        lx = results[y, x-2]
                                            else:
                                                lx = results[y, x-2]
                                        else:
                                            lx = results[y, x-2]
                                else:
                                    lx = results[y, x-2]
                            else:
                                new_label += 1
                                lx = new_label
                                aRTable[lx] = lx
                                aNext[lx] = -1
                                aTail[lx] = lx
                    else:
                        if img[y+1, x+1] == 1:
                            new_label += 1
                            lx = new_label
                            aRTable[lx] = lx
                            aNext[lx] = -1
                            aTail[lx] = lx
                        else:
                            continue

            results[y, x] = lx

    # renumber labels
    cur_label = 0;
    for k in range(1, new_label+1):
        if (aRTable[k]==k):
            cur_label += 1
            aRTable[k] = cur_label
        else:
            aRTable[k] = aRTable[aRTable[k]]

    # second scan
    for y in range(2, h-2, 2):
        for x in range(2, w-2, 2):
            label = results[y, x]
            if label > 0:
                label = aRTable[label]
                if (img[y, x] == 1):
                    results[y, x] = label
                else:
                    results[y, x] = 0

                if (img[y, x+1] == 1):
                    results[y, x+1] = label
                elif results[y, x+1] > 0:
                    results[y, x+1] = 0

                if (img[y+1, x] == 1):
                    results[y+1, x] = label
                elif results[y+1, x] > 0:
                    results[y+1, x] = 0

                if (img[y+1, x+1] == 1):
                    results[y+1, x+1] = label
                elif results[y+1, x+1] > 0:
                    results[y+1, x+1] = 0
            else:
                results[y, x] = 0
                if results[y, x+1] > 0: results[y, x+1] = 0
                if results[y+1, x] > 0: results[y+1, x] = 0
                if results[y+1, x+1] > 0: results[y+1, x+1] = 0

    return cur_label


