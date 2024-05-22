import argparse
import glob
import os
import numpy as np
import json

from coco_classes import COCO_91_CLASSES
from mot_tracker import mot_tracker
from mot_gt import mot_gt

from src import evaluate

# Constants should be in uppercase
OUT_DIR_BASE = 'outputs'
RANDOM_SEED = 42
DEVICE_DEFAULT = 'cuda'


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Script for tracking objects in videos.")
    parser.add_argument('--input_folder', help='Path to folder with MCMT videos.')
    parser.add_argument('--gt_folder', help='Path to folder with MOT-format ground truth (per camera).')
    parser.add_argument('--output_dir', type=str, help='output folder')
    parser.add_argument('--use_roi', action='store_true', help='Whether to use ROI to filter out outliers.')

    parser.add_argument('--config', help='Configuration file.')
    parser.add_argument('--checkpoint', help='Checkpoint file.')
    parser.add_argument('--cls', nargs='+', default=[1], type=int, help='Which classes to track.')
    parser.add_argument('--global_reid_model_name', default='torchreid_osnet_ibn_x1_0_imagenet', help='Type of model to use for global re-ID.')
    parser.add_argument('--reid_model_wts', help='Type of model to use for global re-ID.')

    parser.add_argument('--device', default=DEVICE_DEFAULT, help='Device to use for computation.')
    parser.add_argument('--det_score_agg', action='store_true', help='Whether to aggregate top n reID descriptors.')
    parser.add_argument('--xywh_format', default=True, type=bool, help='gt boxes format')
    parser.add_argument('--use_gt', action='store_true', help='flag to pass gt tracklets for re-ID ')
    return parser.parse_args()


def main():
    """Main function to execute the tracking process."""
    args = parse_arguments()

    np.random.seed(RANDOM_SEED)

    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    exp_number = len(os.listdir(args.output_dir))
    out_dir = os.path.join(args.output_dir, str(exp_number))
    os.makedirs(out_dir, exist_ok=True)

    colors = np.random.randint(0, 255, size=(len(COCO_91_CLASSES), 3))

    print(f"Tracking: {[COCO_91_CLASSES[idx] for idx in args.cls]}")
    print(f"GLOBAL Re-ID embedder: {args.global_reid_model_name}")

    cameras = sorted(glob.glob(os.path.join(args.input_folder, '**', '*.mp4'), recursive=True))
    print(cameras)

    hyp_obj_pairs = []

    for num_camera, camera in enumerate(cameras):
        if args.use_gt:
            pairs = mot_gt(args, camera, out_dir, colors, num_camera)
        else:
            pairs = mot_tracker(args, camera, out_dir, colors, num_camera)
        hyp_obj_pairs.extend(pairs)

    metrics = evaluate(args, out_dir, hyp_obj_pairs)

    print("\n-------------------- Evaluation Metrics: --------------------")
    print(f"Average ID Precision (AP):  {metrics['AP']:.3f}")
    print(f"Average ID Recall (AR):     {metrics['AR']:.3f}")
    print(f"Average ID F1 (AF1):        {metrics['AF1']:.3f}")
    print(f"AUC PR:                     {metrics['AUC_PR']:.3f}")
    print(f"Detector Error:             {metrics['detector_error']:.3f}")
    print(f"ID Switch Error:            {metrics['id_switch_error']:.3f}")
    print(f"ReID Error:                 {metrics['reID_error']:.3f}")
    print("--------------------")
    print('Output dir:', out_dir)
    print(args.config)

    with open(os.path.join(out_dir, "metrics_mcmt.json"), "w") as f:
        json.dump(metrics, f, indent=4)


if __name__ == "__main__":
    main()
