import torch
from torchvision.ops.boxes import nms as torch_nms
from ..utils import count_bbox
"""
0815: Edited out removing background for bbox for FasterRCNN 
"""

class PostProc(object):
    def __init__(self, conf_threshold, nms_threshold, max_boxes, n_classes, coord_h, coord_w):
        self.coord_h, self.coord_w = float(coord_h), float(coord_w)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes
        self.n_classes = n_classes

    def process(self, boxes, probs, img_h, img_w):
        """For Faster-RCNN, boxes has shape # of classes -1  (e.g. COCO: 80, VOC:20) """
        # boxes:    (#boxes, 4) or (#boxes, #classes-1, 4), 4 = ltrbresult_labels = torch.zeros((1)).float().cuda()
        # probs:    (#boxes, #classes+1)

        if len(boxes.shape) == 2:
            # boxes = torch.stack([boxes] * (self.n_classes - 1), dim=1)
            boxes = torch.stack([boxes] * (self.n_classes), dim=1)
        else:
            # boxes = boxes[:, 1:]
            pass

        # recale box coordinate, remove background probability
        # boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]] * (img_w / self.coord_w), min=0, max=img_w - 1)
        # boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]] * (img_h / self.coord_h), min=0, max=img_h - 1)
        boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]] , min=0, max=img_w - 1)
        boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]] , min=0, max=img_h - 1)
        confs = probs[:, :-1] # Remove the last column, which contains the background score
        # boxes:    (#boxes, #classes - 1, 4)
        # confs:    (#boxes, #classes - 1)

        result_boxes = list()
        result_confs = list()
        result_labels = list()

        per_cls_box = []
        for c in range(self.n_classes - 1):
            cls_boxes = boxes[:, c]
            cls_confs = confs[:, c]
            # cls_boxes:    (#boxes, 4)
            # cls_confs:    (#boxes)

            keep_indices = torch.nonzero(cls_confs > self.conf_threshold).view(-1)
            per_cls_box.append(keep_indices.numel())
            if len(keep_indices) == 0:
                continue
            cls_boxes = cls_boxes[keep_indices]
            cls_confs = cls_confs[keep_indices]

            keep_indices = torch_nms(cls_boxes, cls_confs, self.nms_threshold).view(-1)
            if len(keep_indices) == 0:
                continue
            cls_boxes = cls_boxes[keep_indices]
            cls_confs = cls_confs[keep_indices]

            cls_labels = torch.zeros(cls_confs.shape).float().cuda()
            cls_labels += (c)

            result_boxes.append(cls_boxes)
            result_confs.append(cls_confs)
            result_labels.append(cls_labels)

        mean_cnt = sum(per_cls_box) / len(per_cls_box)
        max_cnt = max(per_cls_box)
        count_bbox(mean_cnt,max_cnt)

        if len(result_boxes) > 0:
            result_boxes = torch.cat(result_boxes, dim=0)
            result_confs = torch.cat(result_confs, dim=0)
            result_labels = torch.cat(result_labels, dim=0)
            result_confs, result_ind = result_confs.sort(descending=True)
            result_boxes = result_boxes[result_ind][:self.max_boxes]
            result_confs = result_confs[:self.max_boxes]
            result_labels = result_labels[result_ind][:self.max_boxes]
        else:
            result_boxes = torch.zeros((1, 4)).float().cuda() # Shape : (num_boxes, 4)
            result_confs = torch.zeros((1)).float().cuda()
            result_labels = torch.zeros((1)).float().cuda()
        # result_boxes: (num_boxes, 4)
        # result_onnfs: (num_boxes)
        # result_labels: (num_boxes)
        return result_boxes, result_confs, result_labels