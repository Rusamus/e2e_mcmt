import ast
import glob

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from utils import (
    clusterize,
    convert_tracks,
    get_metric,
    normalize,
    get_pr_curve,
    aggregate_top_n_descriptors,
    extrapolate_until_axes,
    get_errors
)
from bcubed_eval import bcubed_eval


def evaluate(args, output_dir, hyp_obj_pairs):
    """Core evaluation function.

    Parameters:
        output_dir (str): Folder with prediction file per camera in MOT format.
        hyp_obj_pairs (list): Logic:

        - Creates dict1: track_camera -> [bbox1, bbox2, ...]
        - Runs re-ID for each bbox
        - dict2[track_camera] += l2 normalized descriptor * 1/len(track_camera)
        - Matrix similarity -> Clustering (h) -> B-cubed evaluation (ground truth used) -> Returns AP + plot.

    """

    descriptors = []
    index2track = {}
    index = 0

    for num_camera, camera_hyp in enumerate(sorted(glob.glob(f"{output_dir}/*.txt"))):
        df = pd.read_csv(
            camera_hyp,
            names=[
                'fr_num', 'id', 'bbleft', 'bbtop',
                'bbox_width', 'bbox_height', 'det_score',
                'class', 'visibility', 'stuff', 'desc_reid'
            ],
            index_col=False
        )

        # aggregate Re-ID features within traklet id and normalize then
        df['desc_reid'] = df['desc_reid'].apply(lambda x: np.array(ast.literal_eval(x)))
        df = df.sort_values(by=['id', 'det_score'], ascending=[True, False])

        tracklet = df.groupby('id')
        if args.det_score_agg:
            df = tracklet.apply(aggregate_top_n_descriptors)
        else:
            df = tracklet['desc_reid'].agg('mean')

        df = df.apply(lambda x: normalize(x))

        tracks = [f"{x}_c{num_camera}" for x in df.keys()]
        tracks = dict(zip(range(index, index + len(df.keys())), tracks))

        index += len(df.keys())
        index2track.update(tracks)

        descs = np.vstack(df.values.tolist())
        descriptors.append(descs)

    print('Calculating pairwise tracklets distances...')
    descriptors = np.vstack(descriptors)
    dist = pairwise_distances(descriptors, metric='euclidean')

    print('Calculating BCubed metrics...')
    # Loop for h -> clustering -> precision/recall
    precision, recall = [], []
    sample_rate = 1000
    thresholds = np.linspace(dist.min(), dist.max(), sample_rate)

    for threshold in tqdm(thresholds):
        track2cluster = clusterize(dist, threshold, index2track)
        hyp_obj_pairs_h = convert_tracks(hyp_obj_pairs, track2cluster)
        precision_h, recall_h, fpr_h, fnr_h = bcubed_eval(hyp_obj_pairs_h)

        precision.append(precision_h)
        recall.append(recall_h)

    precision, recall = extrapolate_until_axes(precision, recall)  # align with x-y axes
    AP, AR, AF1, auc_pr, reID_error, reID_error_rel = get_metric(precision, recall)
    fp_rate = fpr_h
    fn_rate = fnr_h

    np.savetxt(f"{output_dir}/precision.txt", precision)
    np.savetxt(f"{output_dir}/recall.txt", recall)

    detector_error, id_switch_error = get_errors(precision, recall, fp_rate, fn_rate, auc_pr, reID_error)

    get_pr_curve(precision, recall, auc_pr, detector_error, id_switch_error,
                 reID_error, reID_error_rel,
                 results_path=output_dir)

    metrics = {
        "AP": AP,
        "AR": AR,
        "AF1": AF1,
        "AUC_PR": auc_pr,
        "detector_error": detector_error,
        "id_switch_error": id_switch_error,
        "reID_error": reID_error,
        "reID_error_wrt_upper_bound": reID_error_rel
    }
    return metrics
