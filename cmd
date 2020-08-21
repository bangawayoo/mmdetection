python tools/test.py --config configs/pascal_voc/faster_rcnn_r50_fpn_1x_voc.py \
    --checkpoint checkpoints/faster_rcnn_r50_fpn_1x_voc0712_20200624-c9895d40.pth \
    --eval mAP

python tools/test.py --config configs/pascal_voc/retinanet_r50_fpn_1x_voc0712.py \
    --checkpoint checkpoints/retinanet_r50_fpn_1x_voc0712_20200617-47cbdd0e.pth \
    --eval mAP
