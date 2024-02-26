# flake8: noqa

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torchvision

from torchreid.utils import FeatureExtractor


# Define a function to convert detections to SORT format.
def convert_detections(detections, threshold, classes):
    """Convert detections to SORT format.

    Args:
        detections (dict): Dictionary containing 'boxes', 'labels', and 'scores'.
        threshold (float): Confidence threshold for filtering detections.
        classes (list): List of classes to keep.

    Returns:
        list: List of final detections in SORT format.
    """
    # Get the bounding boxes, labels, and scores from the detections dictionary.
    boxes = detections["boxes"].cpu().numpy()
    labels = detections["labels"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    # Filter out classes not in the specified list.
    lbl_mask = np.isin(labels, classes)
    scores = scores[lbl_mask]

    # Filter out low confidence scores.
    mask = scores > threshold
    boxes = boxes[lbl_mask][mask]
    scores = scores[mask]
    labels = labels[lbl_mask][mask]

    # Convert boxes to [x1, y1, w, h, score] format.
    final_boxes = []
    for box, score, label in zip(boxes, scores, labels):
        final_boxes.append(
            (
                [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                score,
                str(label)
            )
        )

    return final_boxes


def annotate(frame, tracks, colors):
    """Annotate bounding boxes and IDs on the frame.

    Args:
        tracks (list): List of tracks.
        frame (numpy.ndarray): Original frame.
        frame (numpy.ndarray): Resized frame.
        frame_width (int): Width of the original frame.
        frame_height (int): Height of the original frame.
        colors (list): List of colors for bounding box annotation.

    Returns:
        numpy.ndarray: Annotated frame.
    """
 
    for track in tracks:
        track_id = track[0]
        track_class = 1
        x1, y1, x2, y2 = map(int, track[1:5])
        w, h = x2 - x1, y2 - y1
        p1 = (x1, y1)
        p2 = (x2, y2)
        # Annotate boxes.
        color = colors[int(track_class)]
        cv2.rectangle(
            frame,
            p1,
            p2,
            color=(int(color[0]), int(color[1]), int(color[2])),
            thickness=2
        )
        # Annotate ID.
        cv2.putText(
            frame, f"ID: {track_id}",
            (p1[0], p1[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            lineType=cv2.LINE_AA
        )
    return frame



normalize_imgnet = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])


def reid_preprocess(bbox, args):
    """makes reid preprocessing"""

    global_reid_imagenet_norm = args.global_reid_imagenet_norm
    global_reid_warp_size = args.global_reid_warp_size

    resize = torchvision.transforms.Resize(global_reid_warp_size)
    if global_reid_imagenet_norm:
        bbox = normalize_imgnet(bbox)
    bbox = resize(bbox)
    return bbox[None, ...]


def plot_dist_hist(dist_matrix):
    plt.hist(dist_matrix.flatten())
    plt.savefig('dist_hist.png')


def normalize(vector):
    return vector / np.linalg.norm(vector)


def clusterize(matrix_dist, threshold, index2track):
    """Run clustering.

    Parameters:
    matrix_dist (np.array): 2D matrix with distances between local tracks.

    Returns:
    clusters (list): List of sets of global indexes of tracks from clique.
    track2cluster_num (dict): Maps global index of track to cluster index.
    """
    clusters_matrix = np.where(matrix_dist < threshold, 1, 0)
    clusters = []
    track2cluster = {}

    for track_index in range(len(clusters_matrix)):
        nodes = np.nonzero(clusters_matrix[track_index, :])[0]
        clusters = update_clusters_list(clusters, track_index, nodes)

    for num_cluster, cluster in enumerate(clusters):
        for track_index in cluster:
            global_track_name = index2track[track_index]
            track2cluster[global_track_name] = num_cluster

    return track2cluster


def update_clusters_list(clusters, track, nodes):
    """Find clique for curent edges[track,nodes]
    Params
    ------
    clusters: list
        containts list of clusters (sets)
    track:  int
        initial track-node
    nodes:
        other tracks-nodes connected with node

    Returns
    ------
    clusters: list
        list of global indexes of tracks of clique
    """

    tmp_cluster = set([track, *nodes])
    new_nodes = sorted(list(tmp_cluster))
    for node in new_nodes:
        for cluster_set in clusters:
            if node in cluster_set:
                cluster_set.update(tmp_cluster)
                tmp_cluster = cluster_set.copy()
                clusters.remove(cluster_set)
                break

    if tmp_cluster not in clusters:
        clusters.append(tmp_cluster)
    return clusters


def scale_gt_bbox(gt_bboxes, original_hw, target_hw):
    """"applies scaling of gt-bboxes"""

    gt_bboxes.bbleft = gt_bboxes.bbleft.apply(lambda x: int(x * target_hw[1]/original_hw[1]))
    gt_bboxes.bbtop = gt_bboxes.bbtop.apply(lambda x: int(x * target_hw[0]/original_hw[0]))
    gt_bboxes.bbox_width = gt_bboxes.bbox_width.apply(lambda x: int(x * target_hw[1]/original_hw[1]))
    gt_bboxes.bbox_height = gt_bboxes.bbox_height.apply(lambda x: int(x * target_hw[0]/original_hw[0]))

    return gt_bboxes


def get_gt_bboxes(gt_camera, frame_count):
    """look up for gt_bboxes on frame_count frame"""

    gt_bboxes = gt_camera[gt_camera.fr_num == frame_count].iloc[:, 2:6].values
    gt_ids = gt_camera[gt_camera.fr_num == frame_count].iloc[:, 1].values

    return gt_bboxes, gt_ids


def get_box_assigment_list(mota_accumulator, num_camera, filter_flags=['MATCH', 'SWITCH', 'FP', 'MISS']):
    """get list of MOT assignments
    source code is
    https://github.com/cheind/py-motmetrics/blob/beb864a56d6055047c4d6e015188fcc24aca05b7/motmetrics/distances.py#L83

    Params
    ------
    pickle_path: str
        path to pickle MOTAcummulator file
    filter_flags:  list
        flag-words to filter raw MOTAcummulator_events

    Returns
    ------
    hyp_obj_pairs: list
        list of tuples where:
            (i_pred_cluster, j_gt_label) - correct assignment
            (i_pred_cluster, -1) - fp
            (-1, j_gt_label) - fn
    """

    hyp_obj_pairs = []
    for i, pair_status in enumerate(mota_accumulator._events['Type']):
        if pair_status in filter_flags:
            oid = str(mota_accumulator._events['OId'][i])
            hid = str(mota_accumulator._events['HId'][i])

            if np.isnan(float(hid)):
                hid = -1
            else:
                hid = str(int(hid)) + f'_c{num_camera}'

            if np.isnan(float(oid)):
                oid = -1

            hyp_obj_pairs.append((hid, int(oid)))

    return hyp_obj_pairs


def convert_tracks(hyp_obj_pair, track2cluster):
    """maps mot tracks to cluster"""

    track2cluster[-1] = -1
    final_tracks = [(track2cluster[pair[0]], pair[1]) for pair in hyp_obj_pair]
    return final_tracks


def extrapolate_until_axes(precision, recall):
    """Extrapolate precision and recall arrays until axes (0,0) and (1,1)"""
    p = precision.copy()
    r = recall.copy()
    p = [p[0]] + p + [0]
    r = [0] + r + [r[-1]]
    return p, r


def calculate_pr_auc(precision, recall):
    """Calculate PR AUC (area under PR curve)"""
    
    precision, recall = extrapolate_until_axes(precision, recall)
    auc_pr = np.trapz(precision, recall)
    return auc_pr


def save_pr_curve(precisions, recalls, fp_rate, fn_rate, results_path='./outputs', scale_factor=1.5):
    """Save img with precision/recall curve for end-to-end pipeline"""
    path_to_save = os.path.join(results_path, 'MCMT_result.png')
    AUC_PR = calculate_pr_auc(precisions, recalls)


    # Plot PR curve using Seaborn
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8*scale_factor, 6*scale_factor))

    # Plot original precision-recall curve and fill area above it
    sns.lineplot(x=recalls, y=precisions, label=f'End-to-end quality: {AUC_PR:.3f}', color='blue')

    # Get the current axes
    ax = plt.gca()

    # Get the lines from the plot
    lines = ax.get_lines()

    # Extract data points from the curve
    x_data = lines[0].get_xdata()
    y_data = lines[0].get_ydata()

    plt.axhline(y=precisions[0], xmax=recalls[-1], color='black', linestyle='--')
    plt.axhline(y=precisions[0] + fp_rate, xmax=recalls[-1], color='black', linestyle='--')

    detector_error, tracker_error, reID_error = retrieve_errors(precisions, fp_rate, fn_rate, AUC_PR)

    plt.axvline(x=recalls[-1], ymax=precisions[0] + tracker_error, color='black', linestyle='--')

    x, y1, y2  = [0, recalls[-1]], precisions[0], precisions[0] + tracker_error
    plt.fill_between(x, y1, y2, color='yellow', alpha=0.3, label=f'Tracker Error: {tracker_error:.3f}')

    x, y1, y2 = [recalls[-1], 1], 0, 1
    plt.fill_between([recalls[-1], 1], 0, 1, color='red', alpha=0.3, label=f'Detector Error: {detector_error:.3f}')

    x, y1, y2 = [0, recalls[-1]], precisions[0] + fp_rate, 1
    plt.fill_between(x, y1, y2, color='red', alpha=0.3)

    x, y1, y2 = x_data, y_data, y_data[0]
    plt.fill_between(x, y1, y2, color='cyan', alpha=0.3, label=f're-ID Error: {reID_error:.3f}')

    plt.ylabel('Precision', fontsize=12*scale_factor)
    plt.xlabel('Recall', fontsize=12*scale_factor)
    plt.legend(fontsize=18*scale_factor, loc='lower left')
    plt.title('MCMT: Errors Classification', fontsize=16*scale_factor)
    plt.ylim((0, 1.0))
    plt.xlim((0, 1.0))

    plt.savefig(path_to_save, bbox_inches='tight')
    print(path_to_save)

def retrieve_errors(precisions, fp_rate, fn_rate, AUC_PR):
    """makes classification of errors"""

    detector_error = fn_rate + fp_rate
    tracker_error = 1 - precisions[0] - fp_rate
    reID_error = 1 - detector_error - tracker_error - AUC_PR 
    return detector_error, tracker_error, reID_error


def get_metric(precision, recall):
    """Provides final evaluation and returns AP, AR, AF1,
    error_per_module, plot"""
    f1_scores = [2 * (p * r) / (p + r) for p, r in zip(precision, recall)]

    auc_pr = calculate_pr_auc(precision, recall)
    return np.mean(precision), np.mean(recall), np.mean(f1_scores), auc_pr


def get_global_reid_model(args):
    model_name = args.global_reid_model_name
    model_wts_path = os.path.join('reid/weights', f"{args.global_reid_model_wts}.pth")

    global_reid_model = FeatureExtractor(
        model_name=model_name,
        model_path=model_wts_path,
        device='cuda',
    )
    return global_reid_model


def check_is_roi(bbox, roi):
    xmin = bbox[0]
    ymin = bbox[1]
    xmax = bbox[0] + bbox[2]
    ymax = bbox[1] + bbox[3]
    height, width = roi.shape

    if xmin >= 0 and xmin < width:
        if ymin >= 0 and ymin < height and roi[ymin, xmin] == 255:
            return True
        if ymax >= 0 and ymax < height and roi[ymax, xmin] == 255:
            return True
    if xmax >= 0 and xmax < width:
        if ymin >= 0 and ymin < height and roi[ymin, xmax] == 255:
            return True
        if ymax >= 0 and ymax < height and roi[ymax, xmax] == 255:
            return True

    return False

def draw_tracking_results(frame, tracks, colors):
    if len(tracks) > 0:
        frame = annotate(frame, tracks, colors)
    # cv2.putText(
    #     frame,
    #     f"FPS: {fps:.1f}",
    #     (int(20), int(40)),
    #     fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    #     fontScale=1,
    #     color=(0, 0, 255),
    #     thickness=2,
    #     lineType=cv2.LINE_AA
    # )
    return frame