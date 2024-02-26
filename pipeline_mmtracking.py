import mmcv
from mmtrack.apis import inference_mot, init_model
import cv2
import numpy as np
import pandas as pd
import time
import os
import motmetrics as mm
import csv
import torch
from PIL import Image
from utils import (get_gt_bboxes,
                   get_box_assigment_list,
                   get_global_reid_model,
                   draw_tracking_results,
                   check_is_roi)


def camera_tracker(args, video_path, device, out_dir, colors, num_camera, use_roi=False):
    # Initialize mmtracking model
    model = init_model(args.config, args.checkpoint, device=device)

    # re-ID model
    global_reid_model = get_global_reid_model(args)

    # Video stream
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_name = video_path.split(os.path.sep)[-1].split('.')[0]

    reid = args.global_reid_model_wts
    tracker_name = args.config.split('/')[-1].split('.py')[0]
    outfile = f"{out_dir}/{save_name}_{tracker_name}_{reid}.txt"

    # Define codec and create VideoWriter object
    out = cv2.VideoWriter(
        outfile.replace('.txt', '.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (frame_width, frame_height)
    )

    # MOT metrics
    acc = mm.MOTAccumulator()
    mh = mm.metrics.create()

    gt_camera = pd.read_csv(video_path.replace('mp4', 'txt'), names=[
        'fr_num', 'id', 'bbleft', 'bbtop',
        'bbox_width', 'bbox_height', 'det_score',
        'class', 'visibility', 'stuff'],
                        index_col=False)

    if use_roi:
        roi_path = video_path.replace('.mp4', '.roi.png')
        roi_mask = np.asarray(Image.open(roi_path))

    prog_bar = mmcv.ProgressBar(frames)
    with open(outfile, 'w') as file:
        writer = csv.writer(file, delimiter=',')

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to the format expected by mmtrack (BGR -> RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Inference and get tracking results
            result = inference_mot(model, rgb_frame, frame_id=frame_count)
            bboxes = result['track_bboxes'][0]

            # TODO: Adapt the result processing to your needs, e.g., drawing boxes, extracting re-ID features
            gt_bboxes, gt_ids = get_gt_bboxes(gt_camera, frame_count)

            pred_bboxes = []
            pred_ids = []

            # Global reid feature extractor
            for bbox in bboxes:
                x, y, x2, y2 = map(lambda x: abs(int(x)), bbox[1:5])
                w, h = abs(x2 - x), abs(y2 - y)

                if w * h == 0:
                    continue

                if use_roi:
                    if check_is_roi([x, y, w, h], roi_mask) is False:
                        continue

                warp = [rgb_frame[y:y+h, x:x+w, :]]
                descriptor = global_reid_model(warp).detach().cpu().numpy()
                descriptor_str = np.array2string(descriptor.squeeze(), separator=',')

                pred_ids.append(int(bbox[0]))
                pred_bboxes.append([x, y, w, h])

                line = [
                    frame_count, int(bbox[0]),
                    x, y, w, h, -1, -1, -1, -1,
                    descriptor_str
                ]
                writer.writerow(line)

            dists = mm.distances.iou_matrix(gt_bboxes, pred_bboxes)
            acc.update(
                gt_ids,
                pred_ids,
                dists,
                frameid=frame_count)

            # Draw results on the frame (you need to implement this part)
            frame = draw_tracking_results(frame, result['track_bboxes'][0], colors)

            # Write the frame
            out.write(frame)

            # Increment frame count
            frame_count += 1
            prog_bar.update()

    # Release resources
    cap.release()
    out.release()

    summary = mh.compute_many(
        [acc, acc.events.loc[0:1]],
        metrics=mm.metrics.motchallenge_metrics,
        names=['full', 'part'])

    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)

    file_path = "metrics_mot.csv"
    with open(f"{out_dir}/{file_path}", "a") as file:
        file.write(f"Camera {num_camera} Metrics:\n")
        file.write(strsummary + "\n\n")

    all_pair_assigments = get_box_assigment_list(acc, num_camera)
    return all_pair_assigments
