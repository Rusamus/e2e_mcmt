import torch
import os
import argparse
import numpy as np
import glob
import pandas as pd

from coco_classes import COCO_91_CLASSES
# from deepsort import camera_tracker
from pipeline_mmtracking import camera_tracker
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

    parser.add_argument('--config', help='config file')
    parser.add_argument('--checkpoint', help='checkpoint file')

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
    parser.add_argument(
        '--use_roi',
        action='store_true',
        help='whether to use ROI to filter out outliers',
    )
    parser.add_argument(
        '--device',
        default='cuda',
    )

    args = parser.parse_args()

    np.random.seed(42)

    OUT_DIR = 'outputs'
    os.makedirs(OUT_DIR, exist_ok=True)
    exp_num = len(os.listdir(OUT_DIR))
    OUT_DIR += f'/{exp_num}'
    os.makedirs(OUT_DIR, exist_ok=True)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = args.device
    COLORS = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
    # print(f"Detector: {args.model}")
    # print(f"Re-ID embedder: {args.embedder}")
    print(f"GLOBAL Re-ID embedder: {args.global_reid_model_wts}")

    cameras = sorted(glob.glob(args.input_folder + '/**/' + '/*.mp4', recursive=True))
    print(cameras)

    # TODO: run function below in concurrent mode (#proc = #cameras)
    hyp_obj_pairs = []
    
    for num_camera, camera in enumerate(cameras):
        pairs = camera_tracker(args, camera, device, OUT_DIR, COLORS, num_camera, args.use_roi)
        hyp_obj_pairs.extend(pairs)

    evaluate(OUT_DIR, hyp_obj_pairs)
