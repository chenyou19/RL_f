import os

SEED = 42
SHOW_PROGRESS = True
DEBUG_ACTION_TRACE = False

USE_OPENML_CC18 = True
DATASET_SPLIT_MODE = "openml_cc18_holdout"
OPENML_SUITE_ID = 99
OPENML_CACHE_DIR = "data/openml_cache"
TEST_OPENML_TASKS = [
    37,      # diabetes
    53,      # vehicle
    43,      # spambase
    9952,    # phoneme
    9957,    # qsar-biodeg
    146817,  # steel-plates-fault
    3917,    # kc1
    3903,    # pc3
    28,      # optdigits
    32,      # pendigits
    9976,    # madelon
    9910,    # Bioresponse
]

RESULT_DIR = "results"
LOG_DIR = os.path.join(RESULT_DIR, "logs")
FIGURE_DIR = os.path.join(RESULT_DIR, "figures")
TABLE_DIR = os.path.join(RESULT_DIR, "tables")

ACTIONS = [
    "standard_scaler",
    "minmax_scaler",
    "pca",
    "feature_selection",
    "random_forest",
    "svm",
    "knn",
    "evaluate",
]

ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}
ID_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}

MAX_STEPS = 5
INVALID_ACTION_PENALTY = -0.1
STEP_PENALTY = -0.01
PIPELINE_LENGTH_PENALTY = 0.01
INVALID_COUNT_PENALTY = 0.05

STATE_DIM = 12
ACTION_DIM = len(ACTIONS)

EPISODES = 3000
BATCH_SIZE = 30
GAMMA = 0.95
LR = 1e-3
REPLAY_BUFFER_SIZE = 10000
MIN_REPLAY_SIZE = 200
TARGET_UPDATE_FREQ = 20
MODEL_SAVE_FREQ = 100
RESUME_TRAINING = False
RESUME_MODEL_PATH = os.path.join(LOG_DIR, "dqn_agent.pth")

EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 0.995
