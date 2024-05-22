import mmcv
import cv2
import numpy as np
import pandas as pd
import os
import motmetrics as mm
import csv
import warnings
from PIL import Image
import json
from utils import (get_gt_bboxes,
                   get_box_assigment_list,
                   get_global_reid_model,
                   draw_tracking_results,
                   )

from data.preprocess import check_is_roi

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)


def mot_gt(args, video_path, out_dir, colors, num_camera):
    # Initialize mmtracking model
    # pass

    # re-ID model
    global_reid_model = get_global_reid_model(args)

    # Video stream
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_name = video_path.split(os.path.sep)[-1].split('.')[0]
    reid = args.global_reid_model_name
    outfile = f"{out_dir}/{save_name}_gt_{reid}.txt"

    # Define codec and create VideoWriter object
    out = cv2.VideoWriter(
        outfile.replace('.txt', '.mp4'),
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (frame_width, frame_height)
    )

    gt_camera = pd.read_csv(video_path.replace('.mp4', '.txt'),
        names=['fr_num', 'id', 'bbleft', 'bbtop',
        'bbox_width', 'bbox_height', 'det_score',
        'class', 'visibility', 'stuff'],
                        index_col=False)

    if args.use_roi:
        roi_path = os.path.join(os.path.dirname(video_path), 'binary_hull_mask.png')
        roi = np.asarray(Image.open(roi_path))

        meta_path = os.path.join(os.path.dirname(video_path), 'meta.json')
        with open(meta_path, 'r') as file:
            data = json.load(file)

        homography = data['sequence1']['homography'][num_camera]
        offsets = (data['sequence1']['x_offset'], data['sequence1']['y_offset'])

    # MOT metrics
    acc = mm.MOTAccumulator()
    mh = mm.metrics.create()

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

            pred_ids = []
            pred_bboxes = []

            gt_bboxes, gt_ids = get_gt_bboxes(gt_camera, frame_count, args.xywh_format)

            for gt_id, bbox in zip(gt_ids, gt_bboxes):
                x, y, w, h = map(lambda x: abs(int(x)), bbox[:5])
                x2, y2 = abs(x + w), abs(y + h)
                conf = -1

                if w * h == 0:
                    continue

                if args.use_roi:
                    flag = check_is_roi([x, y, x2, y2], homography, roi, offsets)
                    if not flag:
                        continue

                warp = [rgb_frame[y:y+h, x:x+w, :]]

                if warp[0].shape[0] * warp[0].shape[1] == 0:
                    continue

                descriptor = global_reid_model(warp).detach().cpu().numpy()
                descriptor_str = np.array2string(descriptor.squeeze(), separator=',')

                pred_ids.append(int(gt_id))
                pred_bboxes.append([x, y, w, h])

                line = [
                    frame_count, int(gt_id),
                    x, y, w, h, conf, -1, -1, -1,
                    descriptor_str
                ]
                writer.writerow(line)

            pred_ids = gt_ids
            pred_bboxes = gt_bboxes

            dists = mm.distances.iou_matrix(gt_bboxes, pred_bboxes)
            acc.update(
                gt_ids,
                pred_ids,
                dists,
                frameid=frame_count)

            # Draw results on the frame
            frame = draw_tracking_results(frame, pred_bboxes, pred_ids, colors)

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
