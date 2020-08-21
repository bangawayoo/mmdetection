import torch

_g_rpn_timer = []
_g_bbox_counter = {}

def count_bbox(mean, max, rpn=False):
    """
    Input : mean, box
    Counts
        a) Mean number of bboxes per class
        b) Max number of bboxes across class
        c) Image #
    """
    if rpn:
        _g_bbox_counter['image_rpn'] = _g_bbox_counter.get('image_rpn', 0) + 1
        _g_bbox_counter['mean_rpn'] = _g_bbox_counter.get('mean_rpn', 0) + mean
        _g_bbox_counter['max_rpn'] = _g_bbox_counter.get('max_rpn', 0) + max
    else:
        _g_bbox_counter['image'] = _g_bbox_counter.get('image', 0) + 1
        _g_bbox_counter['mean'] = _g_bbox_counter.get('mean', 0) + mean
        _g_bbox_counter['max'] =  _g_bbox_counter.get('max', 0) + max

