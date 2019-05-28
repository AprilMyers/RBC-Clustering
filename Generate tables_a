import glob
import csv
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import Utilities.readin_utils as utils
import imp
from sklearn.decomposition import PCA as sklearnPCA

## Load tables_a
filelist = utils.load_filelist("/home/april/IActData_Export/*")
my_filepath = filelist[0]
my_table, my_voltagelist = utils.load_table(my_filepath, rm_cap=True, scale=False, zero=True)
tables_a = utils.generate_table(filelist, rm_cap=True)
