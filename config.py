import os

SEED = 42
SHOW_PROGRESS = True
DEBUG_ACTION_TRACE = False

USE_OPENML_CC18 = True
DATASET_SPLIT_MODE = "openml_cc18_holdout"
OPENML_SUITE_ID = 99
OPENML_CACHE_DIR = "data/openml_cache"
OPENML_CC18_SUITE_TASKS = [
    3,
    6,
    11,
    12,
    14,
    15,
    16,
    18,
    22,
    23,
    28,
    29,
    31,
    32,
    37,
    43,
    45,
    49,
    53,
    219,
    2074,
    2079,
    3021,
    3022,
    3481,
    3549,
    3560,
    3573,
    3902,
    3903,
    3904,
    3913,
    3917,
    3918,
    7592,
    9910,
    9946,
    9952,
    9957,
    9960,
    9964,
    9971,
    9976,
    9977,
    9978,
    9981,
    9985,
    10093,
    10101,
    14952,
    14954,
    14965,
    14969,
    14970,
    125920,
    125922,
    146195,
    146800,
    146817,
    146819,
    146820,
    146821,
    146822,
    146824,
    146825,
    167119,
    167120,
    167121,
    167124,
    167125,
    167140,
    167141,
]
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
CACHE_DIR = os.path.join(RESULT_DIR, "cache")
PIPELINE_CACHE_PATH = os.path.join(CACHE_DIR, "pipeline_eval_cache.json")
PIPELINE_CACHE_VERSION = "v2"
ENABLE_PIPELINE_CACHE = True

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
INVALID_ACTION_PENALTY = -0.2
STEP_PENALTY = -0.05
PIPELINE_LENGTH_PENALTY = 0.02
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
