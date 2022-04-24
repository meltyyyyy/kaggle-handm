from average_precision import apk
from configs.data import INPUT_DIR
import pandas as pd
import numpy as np


def calculate_apk(list_of_preds, list_of_gts):
    # for fast validation this can be changed to operate on dicts of {'cust_id_int': [art_id_int, ...]}
    # using 'data/val_week_purchases_by_cust.pkl'
    apks = []
    for preds, gt in zip(list_of_preds, list_of_gts):
        apks.append(apk(gt, preds, k=12))
    return np.mean(apks)


def eval_sub(sub_csv, skip_cust_with_no_purchases=True):
    sub = pd.read_csv(sub_csv)
    validation_set = pd.read_feather(INPUT_DIR + 'valid_sample.feather')

    apks = []

    no_purchases_pattern = []
    for pred, gt in zip(sub.prediction.str.split(),
                        validation_set.prediction.str.split()):
        if skip_cust_with_no_purchases and (gt == no_purchases_pattern):
            continue
        apks.append(apk(gt, pred, k=12))
    return np.mean(apks)
