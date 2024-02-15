import os
import time
import csv
import warnings

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import motmetrics as mm

from torchvision.transforms import ToTensor
from deep_sort_realtime.deepsort_tracker import DeepSort
from utils import (convert_detections, annotate, get_gt_bboxes,
                   get_box_assigment_list, get_global_reid_model)

warnings.filterwarnings("ignore")
np.set_printoptions(threshold=np.inf)


def camera_tracker(args, video_path, device, out_dir, colors, num_camera, roi=None):

    # Models
    model = getattr(torchvision.models.detection, args.model)(weights='DEFAULT')
    model.eval().to(device)

    # tracker model
    tracker = DeepSort(max_age=30, embedder=args.embedder, bgr=False,
            embedder_model_name=args.global_reid_model_name,
            embedder_wts=os.path.join('reid/weights', f"{args.global_reid_model_wts}.pth"))

    # re-ID model
    global_reid_model = get_global_reid_model(args)


    # Video stream
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    frame_fps = int(cap.get(5))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    save_name = video_path.split(os.path.sep)[-1].split('.')[0]
    embedder = args.embedder.replace('/', '_')
    reid = args.global_reid_model_wts
    outfile = f"{out_dir}/{save_name}_{args.model}_{embedder}_{reid}.txt"

    # Define codec and create VideoWriter object
    out = cv2.VideoWriter(
        f"{out_dir}/{save_name}_{args.model}_{embedder}.mp4",
        cv2.VideoWriter_fourcc(*'mp4v'), frame_fps,
        (frame_width, frame_height)
    )

    frame_count = 0  # To count total frames
    total_fps = 0  # To get the final frames per second

    # MOT metrics
    acc = mm.MOTAccumulator()
    mh = mm.metrics.create()

    gt_camera = pd.read_csv(video_path.replace('mp4', 'txt'), names=[
        'fr_num', 'id', 'bbleft', 'bbtop',
        'bbox_width', 'bbox_height', 'det_score',
        'class', 'visibility', 'stuff'],
                        index_col=False)

    with open(outfile, 'w') as file:
        writer = csv.writer(file, delimiter=',')

        while cap.isOpened():
            # Read a frame
            ret, frame = cap.read()
            if ret:
                if args.imgsz is not None:
                    resized_frame = cv2.resize(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        (args.imgsz, args.imgsz)
                    )
                else:
                    resized_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Convert frame to tensor and send it to device (cpu or cuda)
                frame_tensor = ToTensor()(resized_frame).to(device)

                start_time = time.time()
                # Feed frame to model and get detections
                det_start_time = time.time()
                with torch.no_grad():
                    detections = model([frame_tensor])[0]
                det_end_time = time.time()

                det_fps = 1 / (det_end_time - det_start_time)

                # Convert detections to Deep SORT format
                detections = convert_detections(detections, args.threshold, args.cls)

                # Update tracker with detections
                track_start_time = time.time()
                tracks = tracker.update_tracks(detections, frame=frame)
                track_end_time = time.time()
                track_fps = 1 / (track_end_time - track_start_time)

                end_time = time.time()
                fps = 1 / (end_time - start_time)
                # Add `fps` to `total_fps`
                total_fps += fps
                # Increment frame count
                frame_count += 1

                print(f"Frame {frame_count}/{frames}",
                      f"Detection FPS: {det_fps:.1f}",
                      f"Tracking FPS: {track_fps:.1f}",
                      f"Total FPS: {fps:.1f}")
                # Draw bounding boxes and labels on frame
                if len(tracks) > 0:
                    frame = annotate(
                        tracks,
                        frame,
                        resized_frame,
                        frame_width,
                        frame_height,
                        colors
                    )
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (int(20), int(40)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=1,
                    color=(0, 0, 255),
                    thickness=2,
                    lineType=cv2.LINE_AA
                )
                out.write(frame)
                if args.show:
                    # Display or save output frame
                    cv2.imshow("Output", frame)
                    # Press q to quit
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                # MOT format

                pred_bboxes = []
                pred_ids = []
                for track in tracks:
                    if track.original_ltwh is not None:
                        x, y, w, h = map(int, track.original_ltwh)

                        # TODO: filter out boxes withing ROI

                        pred_bboxes.append([x, y, w, h])
                        pred_ids.append(track.track_id)

                        # Global reid feature extractor
                        warp = [resized_frame[y:y+h, x:x+w, :]]

                        # Torchreid do the rest of preprocessing
                        descriptor = global_reid_model(warp).detach().cpu().numpy()
                        descriptor = np.array2string(descriptor.squeeze(), separator=',')

                        line = [
                            frame_count, track.track_id,
                            x, y, w, h, -1, -1, -1, -1,
                            descriptor
                        ]
                        writer.writerow(line)

                gt_bboxes, gt_ids = get_gt_bboxes(gt_camera, frame_count)
                # TODO: skip frame if gt aren't provided

                dists = mm.distances.iou_matrix(gt_bboxes, pred_bboxes)
                acc.update(
                    gt_ids,
                    pred_ids,
                    dists,
                    frameid=frame_count)

            else:
                break

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

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    all_pair_assigments = get_box_assigment_list(acc, num_camera)
    return all_pair_assigments
