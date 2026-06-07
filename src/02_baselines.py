# 02_baselines.py
# Run traditional feature selection methods at their best k (found via CV).
# Baselines compete at full strength so the comparison is fair.

import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def load_data():
    with open(C.PROCESSED_PKL, "rb") as f:
        d = pickle.load(f)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"], d["feature_names"]


def eval_subset(X_tr, y_tr, X_te, y_te, mask):
    idx = np.where(mask)[0]
    clf = RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr[:, idx], y_tr)
    preds = clf.predict(X_te[:, idx])
    return accuracy_score(y_te, preds), f1_score(y_te, preds, average="macro", zero_division=0), len(idx)


def cv_best_k(selector_fn, X_tr, y_tr, k_candidates, name):
    print(f"  [{name}] searching k={k_candidates[0]}..{k_candidates[-1]}")
    skf = StratifiedKFold(n_splits=C.CV_FOLDS, shuffle=True, random_state=C.RANDOM_SEED)
    clf = RandomForestClassifier(n_estimators=50, random_state=C.RANDOM_SEED, n_jobs=-1)

    best_k, best_score = k_candidates[0], -1.0
    for k in tqdm(k_candidates, desc=f"    {name}", leave=False):
        sel = selector_fn(k)
        try:
            X_sel = sel.fit_transform(X_tr, y_tr)
        except Exception:
            continue
        score = cross_val_score(clf, X_sel, y_tr, cv=skf, scoring="f1_macro", n_jobs=-1).mean()
        if score > best_score:
            best_score, best_k = score, k
    print(f"    best k={best_k}  CV F1={best_score:.4f}")
    return best_k


def main():
    X_tr, X_te, y_tr, y_te, feat_names = load_data()
    n_feat  = X_tr.shape[1]
    k_cands = [k for k in C.K_CANDIDATES if 1 < k < n_feat]
    results = {}

    print("All features baseline...")
    mask_all = np.ones(n_feat, dtype=bool)
    acc, f1, nf = eval_subset(X_tr, y_tr, X_te, y_te, mask_all)
    results["All Features"] = {"n_features": nf, "accuracy": acc, "macro_f1": f1, "mask": mask_all}
    print(f"  {nf} features  Acc={acc:.4f}  F1={f1:.4f}")

    print("\nMutual Information...")
    best_k = cv_best_k(lambda k: SelectKBest(mutual_info_classif, k=k), X_tr, y_tr, k_cands, "MI")
    sel = SelectKBest(mutual_info_classif, k=best_k).fit(X_tr, y_tr)
    mask = sel.get_support()
    acc, f1, nf = eval_subset(X_tr, y_tr, X_te, y_te, mask)
    results["Mutual Information"] = {"n_features": nf, "accuracy": acc, "macro_f1": f1, "mask": mask, "best_k": best_k}
    print(f"  k={best_k}  Acc={acc:.4f}  F1={f1:.4f}")

    print("\nChi-squared...")
    X_tr_pos = X_tr - X_tr.min(axis=0) + 1e-6
    best_k   = cv_best_k(lambda k: SelectKBest(chi2, k=k), X_tr_pos, y_tr, k_cands, "Chi2")
    sel  = SelectKBest(chi2, k=best_k).fit(X_tr_pos, y_tr)
    mask = sel.get_support()
    acc, f1, nf = eval_subset(X_tr, y_tr, X_te, y_te, mask)
    results["Chi-squared"] = {"n_features": nf, "accuracy": acc, "macro_f1": f1, "mask": mask, "best_k": best_k}
    print(f"  k={best_k}  Acc={acc:.4f}  F1={f1:.4f}")

    print("\nRFE...")
    best_k = cv_best_k(
        lambda k: RFE(RandomForestClassifier(n_estimators=30, random_state=C.RANDOM_SEED, n_jobs=-1), n_features_to_select=k),
        X_tr, y_tr, k_cands, "RFE"
    )
    rfe  = RFE(RandomForestClassifier(n_estimators=30, random_state=C.RANDOM_SEED, n_jobs=-1), n_features_to_select=best_k)
    rfe.fit(X_tr, y_tr)
    mask = rfe.support_
    acc, f1, nf = eval_subset(X_tr, y_tr, X_te, y_te, mask)
    results["RFE"] = {"n_features": nf, "accuracy": acc, "macro_f1": f1, "mask": mask, "best_k": best_k}
    print(f"  k={best_k}  Acc={acc:.4f}  F1={f1:.4f}")

    with open(C.BASELINE_PKL, "wb") as f:
        pickle.dump(results, f)

    print("\nBaseline Results")
    print(f"{'Method':<25} {'k':>5}  {'Accuracy':>10}  {'Macro F1':>10}")
    print("-" * 55)
    for name, r in results.items():
        print(f"{name:<25} {r['n_features']:>5}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")
    print(f"\nSaved to {C.BASELINE_PKL}")


if __name__ == "__main__":
    main()