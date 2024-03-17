_base_ = [
    '/ssd/r.musaev/e2e_mcmt/mmtracking/configs/_base_/models/faster_rcnn_r50_fpn.py',
    '/ssd/r.musaev/e2e_mcmt/mmtracking/configs/_base_/datasets/mot_challenge.py',
    '/ssd/r.musaev/e2e_mcmt/mmtracking/configs/_base_/default_runtime.py'
]

model = dict(
    type='OCSORT',
    detector=dict(
        rpn_head=dict(bbox_coder=dict(clip_border=False)),
        roi_head=dict(
            bbox_head=dict(bbox_coder=dict(clip_border=False), num_classes=1)),
        init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'https://download.openmmlab.com/mmtracking/mot/faster_rcnn/faster-rcnn_r50_fpn_4e_mot17-half-64ee2ed4.pth'  # noqa: E501
        )),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='OCSORTTracker',
        obj_score_thr=0.5,
        reid=dict(
            num_samples=10,
            img_scale=(256, 128),
            img_norm_cfg=None,
            match_score_thr=2.0),
        match_iou_thr=0.5,
        momentums=None,
        num_tentatives=2,
        num_frames_retain=100))
