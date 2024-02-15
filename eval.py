import torch
import os
import argparse
import numpy as np
import glob
import pandas as pd

from coco_classes import COCO_91_CLASSES
from deepsort import camera_tracker
from src import evaluate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_folder',
        help='path to folder with MCMT videos',
    )
    parser.add_argument(
        '--gt_folder', 
        help='path to folder with MOT-format gt (per camera)',
    )
    parser.add_argument(
        '--imgsz',
        default=None,
        help='image resize, 640 will resize images to 640x640',
        type=int
    )
    parser.add_argument(
        '--model',
        default='fasterrcnn_mobilenet_v3_large_fpn',
        help='model name',
        choices=[
            'fasterrcnn_resnet50_fpn_v2',
            'fasterrcnn_resnet50_fpn',
            'fasterrcnn_mobilenet_v3_large_fpn',
            'fasterrcnn_mobilenet_v3_large_320_fpn',
            'fcos_resnet50_fpn',
            'ssd300_vgg16',
            'ssdlite320_mobilenet_v3_large',
            'retinanet_resnet50_fpn',
            'retinanet_resnet50_fpn_v2'
        ]
    )
    parser.add_argument(
        '--threshold',
        default=0.8,
        help='score threshold to filter out detections',
        type=float
    )
    parser.add_argument(
        '--embedder',
        default='mobilenet',
        help='type of feature extractor to use',
        choices=[
            "mobilenet",
            "torchreid",
            "clip_RN50",
            "clip_RN101",
            "clip_RN50x4",
            "clip_RN50x16",
            "clip_ViT-B/32",
            "clip_ViT-B/16"
        ]
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='visualize results in real-time on screen'
    )
    parser.add_argument(
        '--cls',
        nargs='+',
        default=[1],
        help='which classes to track',
        type=int
    )
    parser.add_argument(
        '--global_reid_model_name',
        default='osnet_ibn_x1_0',
        help='type of torchreid model to use for global re-ID',
        choices=[
            "osnet_ibn_x1_0",
        ]
    )
    parser.add_argument(
        '--global_reid_model_wts',
        default='osnet_ibn_x1_0_imagenet',
        help='name of torchreid model weigths',
        choices=[
            "osnet_ibn_x1_0_imagenet",
            "osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter"
        ]
    )
    args = parser.parse_args()

    np.random.seed(42)

    OUT_DIR = 'outputs'
    os.makedirs(OUT_DIR, exist_ok=True)
    exp_num = len(os.listdir(OUT_DIR))
    OUT_DIR += f'/{exp_num}'
    os.makedirs(OUT_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
    print(f"Detector: {args.model}")
    print(f"Re-ID embedder: {args.embedder}")
    print(f"GLOBAL Re-ID embedder: {args.global_reid_model_wts}")

    cameras = sorted(glob.glob(args.input_folder + '/**/' + '/*.mp4', recursive=True))
    print(cameras)

    # TODO: run function below in concurrent mode (#proc = #cameras)
    hyp_obj_pairs = []
    for num_camera, camera in enumerate(cameras):
        pairs = camera_tracker(args, camera, device, OUT_DIR, COLORS, num_camera)
        hyp_obj_pairs.extend(pairs)

    evaluate(OUT_DIR, hyp_obj_pairs)
