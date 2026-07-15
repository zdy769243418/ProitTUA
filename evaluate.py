import torch
import numpy as np


def evaluate_hr(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()

    hit = 0
    for v in rating_k:
        if v in ground_truth:
            hit += 1
    return 1 if hit > 0 else 0


def evaluate_target_attack(rating, ground_truth, top_k, client_id):
    res = [client_id]
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    for v in rating_k:
        if v in ground_truth:
            res.append(v)

    return res


def evaluate_mrr(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()

    for i, v in enumerate(rating_k):
        if v in ground_truth:
            return 1 / (i + 1)

    return 0


def evaluate_ndcg(rating, ground_truth, top_k):
    _, rating_k = torch.topk(rating, top_k)
    rating_k = rating_k.cpu().tolist()
    dcg, idcg = 0., 0.

    for i, v in enumerate(rating_k):
        if i < len(ground_truth):
            idcg += (1 / np.log2(2 + i))
        if v in ground_truth:
            dcg += (1 / np.log2(2 + i))

    return dcg / idcg
