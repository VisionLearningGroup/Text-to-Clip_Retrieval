# --------------------------------------------------------
# Text-to-Clip Retrieval
# Copyright (c) 2019 Boston Univ.
# Licensed under The MIT License [see LICENSE for details]
# by Huijuan Xu
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

def generate_anchors(base_size=8, scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect 
    scales wrt a reference (0, 7) window.
    """

    base_anchor = np.array([1, base_size]) - 1
    anchors = _scale_enum(base_anchor, scales)
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    l = anchor[1] - anchor[0] + 1
    x_ctr = anchor[0] + 0.5 * (l - 1)
    return l, x_ctr 

def _mkanchors(ls, x_ctr):
    """
    Given a vector of lengths (ls) around a center
    (x_ctr), output a set of anchors (windows).
    """

    ls = ls[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ls - 1),
                         x_ctr + 0.5 * (ls - 1)))
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    l, x_ctr = _whctrs(anchor)
    ls = l * scales
    anchors = _mkanchors(ls, x_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    from IPython import embed; embed()
