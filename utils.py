# flake8: noqa

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.transforms import Compose, ToTensor, Resize, Normalize
import torch

from torchreid.utils import FeatureExtractor

np.set_printoptions(threshold=np.inf)


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


def annotate(frame, pred_bboxes, pred_ids, colors):
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
 
    for bbox, track_id in zip(pred_bboxes, pred_ids):
        track_class = 1
        x1, y1, w, h = map(int, bbox)
        x2, y2 = x1 + w, y1 + h
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


def reid_preprocess(bbox, reid_input=[224, 224]):
    """makes reid preprocessing"""
    transforms = Compose([
        ToTensor(),
        Resize(reid_input),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    return transforms(bbox)[None, ...]


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


def get_gt_bboxes(gt_camera, frame_count, xywh_format=True):
    """look up for gt_bboxes on frame_count frame"""

    gt_bboxes = gt_camera[gt_camera.fr_num == frame_count].iloc[:, 2:6].values
    gt_ids = gt_camera[gt_camera.fr_num == frame_count].iloc[:, 1].values

    if not xywh_format:
        gt_bboxes[:, 2:] -= gt_bboxes[:, :2] 


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
    # raise LookupError("NOT CORRECT")

    precision = list(precision)
    recall = list(recall)

    p = [precision[0]] + precision + [0]
    r = [0] + recall + [recall[-1]]
    return p, r


def get_pr_curve(precisions, recalls, auc_pr, detector_error, id_switch_error, reID_error,
                 reID_error_rel, results_path='./outputs', scale_factor=1.5):
    
    """Save img with precision/recall curve for end-to-end pipeline"""
    
    path_to_save = os.path.join(results_path, 'MCMT_result.png')
            
    # Plot PR curve using Seaborn
    sns.set_theme(style="ticks", context="talk")
    plt.figure(figsize=(8*scale_factor, 6*scale_factor))
    plt.style.use('seaborn-v0_8-whitegrid')

    # Plot the step curve for precision-recall
    plt.step(recalls, precisions, where='post', label=f'e2e score: {auc_pr:.3f}', color='blue')

    x_data = []
    y_data = []
    for i in range(len(precisions)-1):
        x_data.extend([recalls[i], recalls[i+1]])
        y_data.extend([precisions[i], precisions[i]])

    x_data.append(recalls[-1])
    y_data.append(precisions[-1])

    x_data = np.array(x_data)
    y_data = np.array(y_data)

    plt.axhline(y=precisions[0], xmax=recalls[-1], color='black', linestyle='--')
    plt.axhline(y=precisions[0] + id_switch_error, xmax=recalls[-1], color='black', linestyle='--')

    plt.axvline(x=recalls[-1], ymax=precisions[0] + id_switch_error, color='black', linestyle='--')
    plt.axvline(x=recalls[-1], ymax=precisions[0] + id_switch_error, color='black', linestyle='--')


    x, y1, y2  = [0, recalls[-1]], precisions[0], precisions[0] + id_switch_error
    plt.fill_between(x, y1, y2, color='yellow', alpha=0.3, label=f'Tracker Error: {id_switch_error:.3f}')


    x, y1, y2 = [recalls[-1], 1], 0, 1
    plt.fill_between([recalls[-1], 1], 0, 1, color='salmon', alpha=0.5, label=f'Detector Error: {detector_error:.3f}')


    x, y1, y2 = [0, recalls[-1]], precisions[0] + id_switch_error, 1
    plt.fill_between(x, y1, y2, color='salmon', alpha=0.5)


    plt.fill_between(x_data, y_data, precisions[0], color='lightsteelblue', alpha=1.0, label=f're-ID Error: {reID_error:.3f}')
    plt.fill_between(x_data, y_data, precisions[0], color='lightsteelblue', alpha=1.0, label=f're-ID Error w.r.t UpperBound: {reID_error_rel:.3f}')

    plt.ylabel('Precision', fontsize=12*scale_factor)
    plt.xlabel('Recall', fontsize=12*scale_factor)
    plt.legend(fontsize=18*scale_factor, loc='lower left')

    plt.title('MCMT: Errors Classification', fontsize=16*scale_factor)
    plt.ylim((0, 1.0))
    plt.xlim((0, 1.0))

    plt.savefig(path_to_save, bbox_inches='tight')
    # plt.savefig(path_to_save.replace('.png', '.pdf'), bbox_inches='tight')
    print(path_to_save)
    

def get_errors(precisions, recalls, fp_rate, fn_rate, auc_pr, reID_error):
    """makes classification of errors"""

    detector_error = fn_rate + fp_rate
    id_switch_error = 1 - precisions[0] - fp_rate

    # det_error = (1 - precisions[0]) * recalls[-1] + (1 - recalls[-1]) * 1
    import sys; sys.breakpointhook()
    assert detector_error + id_switch_error + auc_pr + reID_error == 1

    return detector_error, id_switch_error


def get_metric(precisions, recalls):
    """Provides with final evaluation and returns AP, AR, AF1,
    error_per_module, plot"""
    f1_scores = [2 * (p * r) / (p + r) for p, r in zip(precisions, recalls)]

    auc_pr = np.trapz(precisions, recalls) # approximated area under curve (AUC)
    import sys; sys.breakpointhook()

    reID_error = np.trapz((max(recalls) - np.array(recalls[::-1])), precisions[::-1]) 

    UpperBound_area = precisions[0] * recalls[-1]
    reID_error_rel = reID_error / UpperBound_area

    return np.mean(precisions), np.mean(recalls), np.mean(f1_scores), auc_pr, reID_error, reID_error_rel


def get_global_reid_model(args):

    if 'timm' in args.global_reid_model_name:
        import timm
        global_reid_model = timm.create_model('vit_huge_patch14_224.orig_in21k', pretrained=True, num_classes=0)
        weights = '/ssd/r.musaev/atac/table/timm/vit/mae_vldb_full_100epoch_mae_vit_huge_patch14_64*8batch_0.75mask_70epoch__10epoch_baseline_KD_ensemble_v4_30T_0.85kd_alpha_teacher_head_init_unfreezed_super_ligh_v2_augm_0.05wdDecouple_1e-4lr_v2/experiment.pth'
        state_dict = torch.load(weights, map_location='cpu')
        global_reid_model.load_state_dict(state_dict)
        global_reid_model = global_reid_model.cuda()
        global_reid_model.eval()
        
    elif 'torchreid' in args.global_reid_model_name:
        model_name = args.global_reid_model_name.split('torchreid_')[1]
        global_reid_model = FeatureExtractor(
            model_name=model_name,
            # image_size=(256, 128),
            model_path=args.reid_model_wts if args.reid_model_wts is not None else '',
            device='cuda',
        )
        
    return global_reid_model


def draw_tracking_results(frame, pred_bboxes, pred_ids, colors):
    if len(pred_bboxes) > 0:
        frame = annotate(frame, pred_bboxes, pred_ids, colors)
    return frame


def aggregate_top_n_descriptors(group, top_N=15):
    top_n_descriptors = group.head(top_N)['desc_reid'].tolist()
    mean_descriptor = np.mean(np.stack(top_n_descriptors), axis=0)
    return mean_descriptor