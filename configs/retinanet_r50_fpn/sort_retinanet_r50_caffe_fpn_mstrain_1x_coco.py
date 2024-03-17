_base_ = [
    '/ssd/r.musaev/e2e_mcmt/mmtracking/configs/_base_/models/retinanet_r50_fpn.py',
    '/ssd/r.musaev/e2e_mcmt/mmtracking/configs/_base_/datasets/mot_challenge.py', 
    '/ssd/r.musaev/e2e_mcmt/mmtracking/configs/_base_/default_runtime.py'
]
model = dict(
    type='DeepSORT',

    detector=dict(
        type='RetinaNet',
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmdetection/v2.0/benchmark/retinanet_r50_caffe_fpn_mstrain_1x_coco/retinanet_r50_caffe_fpn_mstrain_1x_coco-586977a0.pth'  # noqa: E501
        )
    ),

    motion=dict(type='KalmanFilter', center_only=False),
    tracker=dict(
        type='SortTracker', obj_score_thr=0.5, match_iou_thr=0.5, reid=None))
