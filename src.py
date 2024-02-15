import ast
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
from tqdm import tqdm

from utils import (
    bcubed_eval,
    clusterize,
    convert_tracks,
    get_metric,
    normalize,
    save_pr_curve
)


def evaluate(output_dir, hyp_obj_pairs):
    """Core evaluation function.

    Parameters:
        output_dir (str): Folder with prediction file per camera in MOT format.
        hyp_obj_pairs (list): Logic:

        - Creates dict1: track_camera -> [bbox1, bbox2, ...]
        - Runs re-ID for each bbox: 
        - dict2[track_camera] += l2 normalized descriptor * 1/len(track_camera)
        - Matrix similarity -> Clustering (h) -> B-cubed evaluation (ground truth used) -> Returns AP + plot.

    """

    descriptors = []
    index2track = {}
    index = 0
    f1_max = 0.0

    df_mcmt = pd.DataFrame(
        columns=[
            'fr_num', 'id', 'bbleft', 'bbtop',
            'bbox_width', 'bbox_height', 'det_score', 'class'
        ]
    )

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

        df_sbst = df[['fr_num', 'id', 'bbleft', 'bbtop',
                      'bbox_width', 'bbox_height', 'det_score', 'class']]

        df_sbst['id'] = df_sbst['id'].apply(lambda x: f"{x}_c{num_camera}")
        df_mcmt = pd.concat([df_mcmt, df_sbst])

        # Re-ID preparation: agg and normalize
        df['desc_reid'] = df['desc_reid'].apply(lambda x: np.array(ast.literal_eval(x)))
        df = df.groupby('id')['desc_reid'].agg('mean')
        df = df.apply(lambda x: normalize(x))

        tracks = [f"{x}_c{num_camera}" for x in df.keys()]
        tracks = dict(zip(range(index, index + len(df.keys())), tracks))

        index += len(df.keys())
        index2track.update(tracks)

        descs = np.vstack(df.values.tolist())
        descriptors.append(descs)

    descriptors = np.vstack(descriptors)
    dist = pairwise_distances(descriptors, metric='euclidean')

    # Loop for h -> clustering -> precision/recall
    precision, recall, fp_rate = [], [], []
    sample_rate = 100
    thresholds = np.linspace(dist.min(), dist.max(), sample_rate)

    for threshold in tqdm(thresholds):
        track2cluster = clusterize(dist, threshold, index2track)
        hyp_obj_pairs_h = convert_tracks(hyp_obj_pairs, track2cluster)
        precision_h, recall_h, fp_rate_h = bcubed_eval(hyp_obj_pairs_h)

        precision.append(precision_h)
        recall.append(recall_h)
        fp_rate.append(fp_rate_h)

        f1_score = 2 * (precision_h * recall_h) / (precision_h + recall_h)
        if f1_score > f1_max:
            f1_max = f1_score
            df_mcmt['id_global'] = df_mcmt['id'].apply(lambda x: track2cluster[x])
            df_mcmt.to_csv(os.path.join(output_dir, "computed_mcmt.csv"), index=False)

    precision, recall = np.array(precision), np.array(recall)
    aidp, aidr, aidf1, auc_pr = get_metric(precision, recall)

    print("\n-------------------- Evaluation Metrics: --------------------")
    print(f"Average ID Precision (AIDP): {aidp:.3f} ")
    print(f"Average ID Recall (AIDR): {aidr:.3f}")
    print(f"Average ID F1 (AIDF1): {aidf1:.3f}")
    print(f"AUC PR: {auc_pr:.3f}")
    print("--------------------")

    metrics = {
        "AIDP": aidp,
        "AIDR": aidr,
        "AIDF1": aidf1,
        "AUC_PR": auc_pr
    }

    with open(os.path.join(output_dir, "metrics_mcmt.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    save_pr_curve(precision, recall, fp_rate, results_path=output_dir)
