import torch
from torchvision.ops.boxes import nms as torch_nms
from ...utils import count_bbox

class PostProc(object):
    def __init__(self, conf_threshold, nms_threshold, max_boxes, n_classes, coord_h, coord_w):
        self.coord_h, self.coord_w = float(coord_h), float(coord_w)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.max_boxes = max_boxes
        self.n_classes = n_classes

    def process(self, boxes, probs, img_h, img_w, score_factors = None):
        # boxes:    (#boxes, 4) or (#boxes, #classes, 4), 4 = ltrbresult_labels = torch.zeros((1)).float().cuda()
        # probs:    (#boxes, #classes)

        if len(boxes.shape) == 2:
            boxes = torch.stack([boxes] * (self.n_classes - 1), dim=1)
        else:
            boxes = boxes[:, 1:]
        # boxes:    (#boxes, #classes - 1, 4)

        # recale box cooridnate, remove background probability
        # boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]] * (img_w / self.coord_w), min=0, max=img_w - 1)
        # boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]] * (img_h / self.coord_h), min=0, max=img_h - 1)
        boxes[:, :, [0, 2]] = torch.clamp(boxes[:, :, [0, 2]] , min=0, max=img_w - 1)
        boxes[:, :, [1, 3]] = torch.clamp(boxes[:, :, [1, 3]] , min=0, max=img_h - 1)
        confs = probs[:, :-1]
        # boxes:    (#boxes, #classes - 1, 4)
        # confs:    (#boxes, #classes - 1)

        result_boxes = list()
        result_confs = list()
        result_labels = list()

        # bbox count
        per_cls_box = []
        # scores for centerness[FCOS]
        # scores = confs * score_factors[:, None]
        for c in range(self.n_classes - 1):
            cls_boxes = boxes[:, c]
            cls_confs = confs[:, c]
            # used in FCOS
            # cls_scores = scores[:, c]
            # cls_boxes:    (#boxes, 4)
            # cls_confs:    (#boxes)

            keep_indices = torch.nonzero(cls_confs > self.conf_threshold).view(-1)
            # bbox count
            per_cls_box.append(keep_indices.numel())
            if len(keep_indices) == 0:
                continue
            cls_boxes = cls_boxes[keep_indices]
            cls_confs = cls_confs[keep_indices]
            # used in fcos
            # cls_confs = cls_scores[keep_indices]


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

        # bbox count
        mean_cnt = sum(per_cls_box) / len(per_cls_box)
        max_cnt = max(per_cls_box)
        count_bbox(mean_cnt, max_cnt)

        if len(result_boxes) > 0:
            # result_boxes = torch.cat(result_boxes, dim=0)[:self.max_boxes]
            # result_confs = torch.cat(result_confs, dim=0)[:self.max_boxes]
            # result_labels = torch.cat(result_labels, dim=0)[:self.max_boxes]
            result_boxes = torch.cat(result_boxes, dim=0)
            result_confs = torch.cat(result_confs, dim=0)
            result_labels = torch.cat(result_labels, dim=0)
            result_confs, result_ind = result_confs.sort(descending=True)
            result_boxes = result_boxes[result_ind][:self.max_boxes]
            result_confs = result_confs[:self.max_boxes]
            result_labels = result_labels[result_ind][:self.max_boxes]
        else:
            result_boxes = torch.zeros((1, 4)).float().cuda()
            result_confs = torch.zeros((1)).float().cuda()
            result_labels = torch.zeros((1)).float().cuda()
        # result_boxes: (#boxes, 4)
        # result_onnfs: (#boxes)
        # result_labels: (#labels)
        return result_boxes, result_confs, result_labels