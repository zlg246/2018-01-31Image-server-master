ver = "4.3.0.2"

"""
This file contains tunable parameters for the image processing algorithms:

 * cropping
 * PL wafer impurity & defect detection
 * G1 grain boundary detection
 * block analysis
 * CZ defect detection
"""

##########
# Server #
##########
HOST = "localhost"
SERVER_PORT = 9900
ONLINE_FF_CORRECTION = False

##########
# WAFERS #
##########

# multi
FAST_MODE = True
INCLUDE_SEMI = False
IMPURE_THRESHOLD = 0.5
BORDER_ERODE = 2
LOWER_CORRECTION = 0.3
MEDIAN_FILTER_ITERATIONS = 0
MASK_SENSITIVITY = 0.2
MIN_FILL_SIZE = 0.0005
MAX_FILL_SIZE = 0.028
DISLOCATION_THRESHOLD = 0.1
DARK_SENSITIVITY = 0.05
DARK_MIN_SIZE = 15
EDGE_THRESH = 4500
STRONG_DIS_THRESHOLD = 0.35
SAW_MARK_MULTI_WAFER = False
SLOPE_MULTI_WAFER = False
MIN_IMPURE_AREA = 0.001

# cropping
EDGE_RATIO = 1.25
WAFER_CROP_TOLERANCE = 4

# robust dislocations
DOG_STRUCT_SIZE = 7
DOG_SIGMA1 = 1.0
DOG_SIGMA2 = 27.5
DOG_THRESH = 0.06
ROBUST_CROP = 25
ROBUST_METRIC_UPPER = 40.0
ROBUST_METRIC_POWER = 1.0

# G1
#RIPPLE_RATIO_THRESHOLD = 1.08
#EDGE_THRESH_RIPPLE = 2000
#GBD_BORDER = BORDER_ERODE


# mono
MAX_CRACK_AREA = 0.01
MIN_CZ_CRACK_SIZE = 10
MONO_WAFER_HISTOGRAM_ONLY = False
DETECT_SLIP_LINES = False
DETECT_STRIATIONS = False
MIN_SLIP_NUM = 2
MIN_SLIP_DIST = 300
STRIPE_CORRECTION = 3 # {0: None, 1: Simple, 2: Skewed, 3: Rounded}
BORDER_ERODE_CZ = -1
MAX_NUM_CRACKS = 30
CRACK_DETAILS = 5
MONO_CROP_SNR = 2



##########
# BLOCKS #
##########

BLOCK_DISLOCATION_THRESH = 0.075 # 0.15

CUTTING_THRESH_BOTTOM_SP = 0.31
CUTTING_THRESH_TOP_SP = 0.6
CUTTING_THRESH_BOTTOM_PLIR = 0.682 # 0.7
CUTTING_THRESH_TOP_PLIR = 0.928 # 1.0
BRICK_EDGE_THRESH = 0.5

PLIR_INTERPOLATE_MARKER_WIDTH = 9 # 0 to turn this off

#########
# CELLS #
#########

# general
CELL_EDGE_THRESH = 0.023
CELL_EDGE_STD_THRESH = 0.25
CELL_NO_BBS = False
CELL_BB_MID_POINTS = False

# cropping
CELL_BORDER_ERODE = 0
CELL_BACKGROUND_THRESH = 0.3
CELL_DIFF_THRESH = 0.1

# 1=variation based (recommended)
# 2=intensity based
# 3=hard coded (use CELL_BB_WIDTH)
CELL_BB_WIDTH_MODE = 1
CELL_BB_WIDTH = 10
CELL_BB_THRESH = 0.5
ORIGINAL_ORIENTATION = True
CELL_BB_MIN = 0.4
CELL_BB_MODE_2_STD = 4.4

# MONO

# broken fingers
BROKEN_FINGER_EDGE_THRESHOLD = 0.7
BROKEN_FINGER_THRESHOLD2 = 1.0
BROKEN_FINGER_THRESHOLD1 = 2

# firing (splotches)
FIRING_SENSITIVITY = 0.0

# bright areas
BRIGHT_AREA_SENSITIVITY = -0.0

# dark spots
DARK_SPOT_MIN_SIZE = 3
DARK_SPOT_MIN_STRENGTH = 0.1

# dark areas
DARK_AREA_SENSITIVITY = 0.0

# crack detection
CELL_CRACK_STRENGTH = 5.4

# dark middles
CELL_DARK_MIDDLE_MIN = 0.0
CELL_DARK_MIDDLE_SCALE = 1.0

# MULTI

# broken fingers
BROKEN_FINGER_MULTI_THRESHOLD1 = 0.25 #0.4
BROKEN_FINGER_MULTI_EDGE_THRESHOLD = 0.7

# bright lines
BRIGHT_LINES_MULTI_SENSITIVITY = 0.5
BRIGHT_LINES_MULTI_THRESHOLD = 2.5

# CELL EFFICIENCY
CELL_DISLOCATION_SENSITIVITY = 0.0

# bright areas
BRIGHT_AREA_MULTI_SENSITIVITY = -0.0

# parameters for anisotropic diffusion
ANISO_DIFF_NUM_ITERATION = 25
ANISO_DIFF_KAPPA = 10
ANISO_DIFF_GAMMA = 0.1

# factor for image binarization
IMG_BIN_INTENSITY_FACTOR = 2

###################################
# parameters for connected region #
##################################

# min size (i.e., size = # of image rows/CONN_REGION_MIN_SIZE_FACTOR)
CONN_REGION_MIN_SIZE_FACTOR = 5.0
# min mean intensity
CONN_REGION_MIN_MEAN_INTENSITY = 0.9
# min width/height
CONN_REGION_MIN_WID_HEI = 10
# min std. of pixel distances to left & bottom point.
CONN_REGION_MIN_DISTANCE_STD = 1.45
# max eccentricity
CONN_REGION_MAX_ECCENTRICITY = 0.97
# max intensity difference with surrounding region
CONN_REGION_MAX_INTENSITY_DIFF = 0.06
# max intensity ratio with surrounding region
CONN_REGION_MAX_INTENSITY_RATIO = 0.12
# max mean intensity of surrounding region.
SURR_REGION_MAX_MEAN_INTENSITY = 0.27


########
# SLUG #
########

RING_SIGMA1 = 0.5
RING_SIGMA2 = 5
NUM_PEAKS = 3

###########
# STRIPES #
###########

STRIPE_BACKGROUND_THRESH = 0.2


######################
# CALLBACK FUNCTIONS #
######################

def __FullyImpure(features): return None
def __WaferClassify(features): return
def __Preprocess(im, features): return im
