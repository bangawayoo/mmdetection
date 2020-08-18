import torch


_g_bbox_counter = {}

def count_bbox(mean, max):
    """
    Input : mean, box
    Counts
        a) Mean number of bboxes per class
        b) Max number of bboxes across class
        c) Image #
    """
    _g_bbox_counter['image'] = _g_bbox_counter.get('image', 0) + 1
    _g_bbox_counter['mean'] = _g_bbox_counter.get('mean', 0) + mean
    _g_bbox_counter['max'] =  _g_bbox_counter.get('max', 0) + max


