from collections import Counter, defaultdict


def bcubed_eval(pairs):
    """
    Clusters evaluation based on https://link.springer.com/article/10.1007/s10791-008-9066-8

    Params
    ------
    pairs: list
        List of tuples where:
            (i_pred_cluster, j_gt_label) - correct assignment
            (i_pred_cluster, -1) - false positive
            (-1, j_gt_label) - false negative

    Returns
    ------
    (mean_p, mean_r, mean_fp, mean_fn): tuple
        Mean (precision, recall, fpr, fnr) per threshold
    """
    h2o = defaultdict(list)
    o2h = defaultdict(list)

    for pair in pairs:
        h2o[pair[0]].append(pair[1])
        o2h[pair[1]].append(pair[0])

    P, R, FP, FN = 0, 0, 0, 0
    FREQ_pred, FREQ_gt = 0, 0

    for key in h2o:
        if key == -1:
            FN += len(h2o[key])
            continue

        cluster = Counter(h2o[key])
        cluster_size = len(h2o[key])

        for item, freq in cluster.items():
            if item == -1:
                FREQ_pred += freq
                continue

            P += freq * freq / cluster_size
            FREQ_pred += freq

    mean_p = P / FREQ_pred

    for key in o2h:
        if key == -1:
            FP += len(o2h[key])
            continue

        cluster = Counter(o2h[key])
        cluster_size = len(o2h[key])

        for item, freq in cluster.items():
            if item == -1:
                FREQ_gt += freq
                continue

            R += freq * freq / cluster_size
            FREQ_gt += freq

    mean_r = R / FREQ_gt
    mean_fp_rate = FP / FREQ_pred
    mean_fn_rate = FN / FREQ_gt

    return mean_p, mean_r, mean_fp_rate, mean_fn_rate
