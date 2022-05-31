from sklearn.feature_extraction import text

STOP_WORDS = list(text.ENGLISH_STOP_WORDS)
STOP_WORDS.remove('us')

MAX_MISSING_PERCENTAGE = 30
MISSING_VALUES = ['N/A', 'N/a', 'n/a', 'NULL', 'Null', 'null', 'Nan', 'nan', 'None', 'none', '?', '-', ' ']

CONTINUOUS_THRESHOLD = 0.002
MIN_UNIQUE_CONTINUOUS_VALUES = 5
MAX_PERCENTAGE_UNIQUE_OUTLIERS = 30
CATEGORICAL_OUTLIERS_DETECTION_THRESHOLD = 0.1
WO_OUTLIERS_DET = ['_is_missing', '_chars_no', '_avg_word_len']

PRIORITY = 5
VERBOSE = False
META_MODEL_TYPE = 0
EARLY_STOPPING_BO = 0
META_MODEL_PROB_TYPE = 0
ITERATION_META_PROB = None
TIME_LIMITED_OPTIMIZATION = True

log_text_1 = ''
log_text_2 = ''
bayesian_opt_ = None
linear_model_ = False
initial_opt_time = 0.0
min_models_opt_iter = None
max_optimization_time = None
best_scores_optimization = []
max_optimization_time_mlpc = None
max_model_optimization_time = None
