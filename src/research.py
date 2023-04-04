import os
import sys
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("src/")
import utils

data = utils.read_dmp_data("data/dmp_loc_traces_Feb10to28_sample100IDs.csv")
