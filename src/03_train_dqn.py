# 03_train_dqn.py - train Double DQN agent for feature selection on UNSW-NB15

import os, sys, pickle, warnings
import numpy as np
import torch

warnings.filterwarnings("ignore")

from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
from env import FeatureSelectionEnv
from agent import DoubleDQNAgent


def load_data():
    with open(C.PROCESSED_PKL, "rb") as f:
        d = pickle.load(f)
    return d["X_train"], d["X_test"], d["y_train"], d["y_test"], d["feature_names"]


def main():
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    X_tr, X_te, y_tr, y_te, feat_names = load_data()
    n_feat = X_tr.shape[1]
    print(f"features={n_feat}  train={X_tr.shape[0]}  episodes={C.NUM_EPISODES}")

    env   = FeatureSelectionEnv(X_tr, y_tr)
    agent = DoubleDQNAgent(state_size=n_feat, action_size=n_feat, device=torch.device("cpu"))

    metrics = {
        "episode_rewards": [],
        "episode_f1":      [],
        "episode_nfeat":   [],
        "episode_epsilon": [],
    }

    best_f1_seen   = -1.0
    best_mask_seen = np.ones(n_feat, dtype=bool)
    best_episode   = 0

    eval_clf = LogisticRegression(max_iter=100, solver="lbfgs", random_state=C.RANDOM_SEED)

    for ep in tqdm(range(1, C.NUM_EPISODES + 1), desc="training"):
        state     = env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store(state, action, reward, next_state, done)
            if ep > C.WARMUP_EPISODES:
                agent.learn()
            state      = next_state
            ep_reward += reward

        agent.end_episode(ep)

        mask  = state.astype(bool)
        n_sel = int(mask.sum())
        idx   = np.where(mask)[0]
        ep_f1 = 0.0
        if n_sel > 0:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    eval_clf.fit(env.X_tr[:, idx], env.y_tr)
                    preds = eval_clf.predict(env.X_val[:, idx])
                ep_f1 = f1_score(env.y_val, preds, average="macro", zero_division=0)
            except Exception:
                pass

        metrics["episode_rewards"].append(ep_reward)
        metrics["episode_f1"].append(ep_f1)
        metrics["episode_nfeat"].append(n_sel)
        metrics["episode_epsilon"].append(agent.epsilon)

        if ep_f1 > best_f1_seen:
            best_f1_seen   = ep_f1
            best_mask_seen = mask.copy()
            best_episode   = ep

        if ep % 50 == 0:
            avg_r  = np.mean(metrics["episode_rewards"][-50:])
            avg_f1 = np.mean(metrics["episode_f1"][-50:])
            avg_nf = np.mean(metrics["episode_nfeat"][-50:])
            print(
                f"  ep={ep}  eps={agent.epsilon:.3f}  "
                f"avg_reward={avg_r:.4f}  avg_f1={avg_f1:.4f}  "
                f"avg_features={avg_nf:.1f}  best_f1={best_f1_seen:.4f} (ep {best_episode})"
            )

    agent.save(C.MODEL_PATH)

    best_result = {
        "mask":          best_mask_seen,
        "val_f1":        best_f1_seen,
        "n_features":    int(best_mask_seen.sum()),
        "episode":       best_episode,
        "feature_names": [feat_names[i] for i in np.where(best_mask_seen)[0]],
    }
    with open(C.DQN_BEST_PKL, "wb") as f:
        pickle.dump(best_result, f)
    with open(C.METRICS_PATH, "wb") as f:
        pickle.dump(metrics, f)

    print(f"\ntraining done")
    print(f"  best val F1 : {best_f1_seen:.4f}  (episode {best_episode})")
    print(f"  features    : {int(best_mask_seen.sum())} / {n_feat}")
    print(f"  selected    : {best_result['feature_names']}")


if __name__ == "__main__":
    main()