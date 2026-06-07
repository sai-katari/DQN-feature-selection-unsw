# 01_preprocess.py
# Load UNSW-NB15 train/test CSVs, clean, normalize, save to processed.pkl
# Uses the official benchmark split so results match published papers.

import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

TRAIN_CSV = os.path.join(C.DATA_DIR, "UNSW_NB15_training-set.csv")
TEST_CSV  = os.path.join(C.DATA_DIR, "UNSW_NB15_testing-set.csv")

DROP_COLS = ["id", "attack_cat"]


def load_and_check():
    missing = [os.path.basename(f) for f in [TRAIN_CSV, TEST_CSV] if not os.path.exists(f)]
    if missing:
        print("Missing files in data/:", missing)
        print("Download from the UNSW-NB15 dataset page and place in data/")
        sys.exit(1)


def clean(df, scaler=None, fit=False):
    drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop)

    if C.LABEL_COL not in df.columns:
        df.rename(columns={df.columns[-1]: C.LABEL_COL}, inplace=True)

    y    = df[C.LABEL_COL].copy()
    X_df = df.drop(columns=[C.LABEL_COL])

    # encode categoricals
    cat_cols = X_df.select_dtypes(include=["object", "category"]).columns.tolist()
    le = LabelEncoder()
    for c in cat_cols:
        X_df[c] = le.fit_transform(X_df[c].astype(str))

    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    if fit:
        X_df = X_df.fillna(X_df.median())
    else:
        X_df = X_df.fillna(0)

    y = (pd.to_numeric(y, errors="coerce").fillna(0) > 0).astype(int)

    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(np.float32)

    if fit:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    return X, y.values, feature_names, scaler


def main():
    os.makedirs(C.DATA_DIR, exist_ok=True)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)

    load_and_check()

    print("Loading training set...")
    df_train = pd.read_csv(TRAIN_CSV)
    print(f"  shape: {df_train.shape}")

    print("Loading test set...")
    df_test = pd.read_csv(TEST_CSV)
    print(f"  shape: {df_test.shape}")

    print("Cleaning...")
    X_tr, y_tr, feat_names, scaler = clean(df_train, fit=True)
    X_te, y_te, _, _               = clean(df_test, scaler=scaler, fit=False)

    print(f"  X_train: {X_tr.shape}  X_test: {X_te.shape}")
    print(f"  Features: {feat_names}")

    payload = {
        "X_train":       X_tr,
        "X_test":        X_te,
        "y_train":       y_tr,
        "y_test":        y_te,
        "feature_names": feat_names,
        "n_features":    X_tr.shape[1],
        "scaler":        scaler,
    }
    with open(C.PROCESSED_PKL, "wb") as f:
        pickle.dump(payload, f)

    print(f"Saved to {C.PROCESSED_PKL}")


if __name__ == "__main__":
    main()