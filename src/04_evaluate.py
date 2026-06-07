# 04_evaluate.py
# Evaluate DQN best subset vs baselines on the held-out test set.

import os
import sys
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


def eval_on_test(X_tr, y_tr, X_te, y_te, mask):
    idx = np.where(mask)[0]
    clf = RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr[:, idx], y_tr)
    preds = clf.predict(X_te[:, idx])
    return accuracy_score(y_te, preds), f1_score(y_te, preds, average="macro", zero_division=0), clf, preds


def main():
    with open(C.PROCESSED_PKL, "rb") as f:
        data = pickle.load(f)
    X_tr, X_te     = data["X_train"], data["X_test"]
    y_tr, y_te     = data["y_train"], data["y_test"]
    feat_names     = data["feature_names"]
    n_feat         = X_tr.shape[1]

    with open(C.BASELINE_PKL, "rb") as f:
        baselines = pickle.load(f)

    with open(C.DQN_BEST_PKL, "rb") as f:
        dqn_best = pickle.load(f)

    final_results = {}

    print(f"\n{'Method':<26} {'k':>5}  {'Accuracy':>10}  {'Macro F1':>10}")
    print("-" * 58)

    for name, bres in baselines.items():
        acc, f1, _, _ = eval_on_test(X_tr, y_tr, X_te, y_te, bres["mask"])
        final_results[name] = {"n_features": bres["n_features"], "accuracy": acc, "macro_f1": f1}
        print(f"{name:<26} {bres['n_features']:>5}  {acc:>10.4f}  {f1:>10.4f}")

    dqn_acc, dqn_f1, _, dqn_preds = eval_on_test(X_tr, y_tr, X_te, y_te, dqn_best["mask"])
    dqn_nf = int(dqn_best["mask"].sum())
    final_results["DQN Agent"] = {"n_features": dqn_nf, "accuracy": dqn_acc, "macro_f1": dqn_f1}

    print("-" * 58)
    print(f"{'DQN Agent':<26} {dqn_nf:>5}  {dqn_acc:>10.4f}  {dqn_f1:>10.4f}")

    all_f1   = baselines["All Features"]["macro_f1"]
    pct_saved = 100 * (n_feat - dqn_nf) / n_feat
    print(f"\nDQN selected {dqn_nf}/{n_feat} features ({pct_saved:.1f}% reduction)")
    print(f"F1 vs all features: {dqn_f1 - all_f1:+.4f}")

    print("\nClassification Report (DQN subset):")
    print(classification_report(y_te, dqn_preds, target_names=["Normal", "Attack"]))

    selected = [feat_names[i] for i in np.where(dqn_best["mask"])[0]]
    print(f"Selected features ({dqn_nf}):")
    for i, name in enumerate(selected, 1):
        print(f"  {i:2d}. {name}")

    with open(os.path.join(C.RESULTS_DIR, "final_results.pkl"), "wb") as f:
        pickle.dump(final_results, f)

    summary = [
        "DQN Feature Selection on UNSW-NB15 - Results",
        f"{'Method':<26} {'k':>5}  {'Accuracy':>10}  {'Macro F1':>10}",
        "-" * 55,
    ]
    for name, r in final_results.items():
        summary.append(f"{name:<26} {r['n_features']:>5}  {r['accuracy']:>10.4f}  {r['macro_f1']:>10.4f}")
    summary.append(f"\nDQN: {dqn_nf} features ({pct_saved:.1f}% reduction), F1 delta: {dqn_f1 - all_f1:+.4f}")

    with open(os.path.join(C.RESULTS_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(summary))

    print(f"\nSaved to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()