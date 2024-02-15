# Multi-Camera Multi-Target Tracking System

This repository contains code for a multi-camera multi-target tracking (MCMT) system using object detection, tracking, and re-identification.

## System Overview

The system pipeline consists of:

- Object detection - Detects objects in each camera frame using a RetinaNet model
- Tracking - Associates detections into tracks for each camera using DeepSORT 
- Re-identification - Embeds tracklet features using a torchreid model and matches tracklets across cameras

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/mcmt-tracking.git
```

Install dependencies:

```bash 
pip install -r requirements.txt
```

Download pretrained models from:

- [RetinaNet model](https://example.com)
- [DeepSORT model](https://example.com)
- [Torchreid model](https://example.com) 

Place the models in the `models/` directory.

## Usage

To run tracking on a dataset:

```bash
CUDA_VISIBLE_DEVICES=0 python3 eval.py --input_folder /path/to/dataset --mode retinanet_resnet50_fpn_v2 --embedder torchreid --global_reid_model_wts osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter --cls 1 2 3  
```

This will perform detection, tracking, re-id and output tracking results to `outputs/`.  

See `options.py` for additional parameters.

## Evaluation

Tracking results can be evaluated using:  

```bash
python evaluate.py --groundtruths /path/to/groundtruths --results /path/to/results  
```

Evaluation code computes ID measures like ID Precision, ID Recall, ID F1 Score.

## References

Ristani et al. "Performance measures and a data set for multi-target, multi-camera tracking", ECCV 2016.

## Citation

```
@article{mypaper,
  title={My Method for Multi-Camera Multi-Target Tracking}, 
  author={My Name},
  journal={arXiv},
  year={2024}
}
```

Let me know if you have any other questions!