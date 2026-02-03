import sys, os
import pdb
sys.path.append(os.path.join(os.path.dirname("__file__"), '..'))
sys.path.append(os.path.join(os.path.dirname("__file__"), '..', '..'))
pos='snap'
path = os.getcwd()
SRC_PATH = path.split('standard_repo')[0]+"/standard_repo"
CURRENT_WP=os.path.dirname(SRC_PATH)
PARENT_WP=os.path.dirname(CURRENT_WP)
if pos=='snap':
    EXP_PATH = CURRENT_WP
else:
    raise ValueError("Please specify the position of the project directory")
