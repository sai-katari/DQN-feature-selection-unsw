# config.py - hyperparameters and paths

# Paths
DATA_DIR      = "data/"
RESULTS_DIR   = "results/"
PROCESSED_PKL = "data/processed.pkl"
BASELINE_PKL  = "results/baselines.pkl"
DQN_BEST_PKL  = "results/dqn_best.pkl"
MODEL_PATH    = "results/dqn_model.pt"
METRICS_PATH  = "results/training_metrics.pkl"

# Dataset
LABEL_COL    = "label"
TEST_SIZE    = 0.20
RANDOM_SEED  = 42

# RL environment
TOTAL_FEATURES   = 42
EPISODE_STEPS    = 50
MIN_FEATURES     = 4
LAMBDA_SPARSITY  = 0.05

# DQN architecture
HIDDEN_SIZES     = [128, 128, 64]
LEARNING_RATE    = 1e-3
GAMMA            = 0.99
BATCH_SIZE       = 64
REPLAY_BUFFER    = 10_000
TARGET_UPDATE_EP = 10

# Exploration
EPS_START    = 1.0
EPS_END      = 0.01
EPS_DECAY_EP = 200

# Training
NUM_EPISODES    = 300
WARMUP_EPISODES = 20

# Baselines
VAL_FRACTION = 0.15
K_CANDIDATES = list(range(5, 42, 3))
CV_FOLDS     = 5