# env.py - feature selection environment
# State: binary vector of length n_features (1=selected, 0=dropped)
# Action: toggle one feature on/off
# Reward: validation F1 minus sparsity penalty

import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C


class FeatureSelectionEnv:
    def __init__(self, X_train, y_train):
        # subsample for reward speed - 5k train, 2k val is enough signal
        sss = StratifiedShuffleSplit(n_splits=1, train_size=5000, random_state=C.RANDOM_SEED)
        idx_tr, idx_val = next(sss.split(X_train, y_train))
        self.X_tr  = X_train[idx_tr]
        self.y_tr  = y_train[idx_tr]
        self.X_val = X_train[idx_val[:2000]]
        self.y_val = y_train[idx_val[:2000]]

        self.n_features = self.X_tr.shape[1]
        self.n_actions  = self.n_features

        self._clf = LogisticRegression(max_iter=100, solver="lbfgs", random_state=C.RANDOM_SEED)

        self.state    = None
        self.step_cnt = None
        self.reset()

    def reset(self):
        # start each episode with all features selected
        self.state    = np.ones(self.n_features, dtype=np.float32)
        self.step_cnt = 0
        return self.state.copy()

    def step(self, action):
        assert 0 <= action < self.n_actions

        new_state = self.state.copy()
        new_state[action] = 1.0 - new_state[action]

        # reject toggle if it would drop below minimum feature count
        if int(new_state.sum()) < C.MIN_FEATURES:
            reward = -0.01
        else:
            self.state = new_state
            reward = self._compute_reward(self.state)

        self.step_cnt += 1
        done = self.step_cnt >= C.EPISODE_STEPS

        info = {
            "n_selected": int(self.state.sum()),
            "selected_indices": np.where(self.state)[0].tolist(),
        }
        return self.state.copy(), reward, done, info

    def observation_space_shape(self):
        return (self.n_features,)

    def action_space_size(self):
        return self.n_actions

    def _compute_reward(self, state):
        idx = np.where(state)[0]
        if len(idx) == 0:
            return -1.0
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._clf.fit(self.X_tr[:, idx], self.y_tr)
                preds  = self._clf.predict(self.X_val[:, idx])
            f1_val = f1_score(self.y_val, preds, average="macro", zero_division=0)
        except Exception:
            return -0.5

        sparsity_penalty = C.LAMBDA_SPARSITY * (len(idx) / self.n_features)
        return float(f1_val - sparsity_penalty)

    def evaluate_subset(self, mask, X_test, y_test):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return 0.0, 0.0
        clf = RandomForestClassifier(n_estimators=100, random_state=C.RANDOM_SEED, n_jobs=-1)
        clf.fit(self.X_tr[:, idx], self.y_tr)
        preds = clf.predict(X_test[:, idx])
        return accuracy_score(y_test, preds), f1_score(y_test, preds, average="macro", zero_division=0)