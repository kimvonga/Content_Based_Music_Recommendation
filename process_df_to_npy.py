import numpy as np
import pandas as pd
import sys
import os
from scipy.interpolate import Akima1DInterpolator
from sklearn.feature_extraction import DictVectorizer
from optparse import OptionParser

'''
Goal is to process pandas dataframe, which contains raw data from .h5 files,
into numpy arrays for training. The output of this file will be pitch, timbre,
features1D, and labels, all as .npy files
'''

parser = OptionParser()
parser.add_option('-i','--input',dest='input_df',help='input pandas df as .pkl file',default='./')
parser.add_option('-o','--o_dir',dest='o_dir',help='output directory',default='./')

(options, args) = parser.parse_args()
input_df = options.input_df
o_dir = options.o_dir

# Check if files already exist and stop if they do
if input_df[-8]+'pitch.npy' in os.listdir(o_dir) and input_df[-8]+'timbre.npy' in os.listdir(o_dir):
    if input_df[-8]+'features1D.npy' in os.listdir(o_dir) and input_df[-8]+'labels.npy' in os.listdir(o_dir):
        sys.exit('npy files for subset '+input_df[-8]+' already exist')

# Load the dataset for processing
subset = pd.read_pickle(input_df)

# Figuring out which indices to drop
seg_lens = [len(x) for x in subset['segments_start']]
ind_short = [ind for ind,val in enumerate(seg_lens) if val < 250]
ind_long = [ind for ind,val in enumerate(seg_lens) if val > 2000]

terms80 = np.load('A_terms80.npy')
red_terms = []
for i in range(subset.shape[0]):
    mask = [True if x in terms80 else False for x in subset['artist_terms'][i]]
    red_terms.append(subset['artist_terms'][i][mask])
ind_null = np.argwhere(np.array([len(x) for x in pd.Series(red_terms)]) == 0).reshape(-1).tolist()

drop_ind = set(ind_short+ind_long+ind_null)

# Dropping indices
subset.drop(index=drop_ind, inplace=True)
subset.reset_index(inplace=True, drop=True)

# Processing 2D features (pitch, timbre) first
pitch, timbre = [], []
for ind, seg in enumerate(subset['segments_start']):
    li_li_pitch, li_li_timbre = np.zeros([1000, 12]), np.zeros([1000,12])
    start, end = seg[0], seg[-1]
    for j in range(12):
        li_li_pitch[:,j] = Akima1DInterpolator(seg, subset['segments_pitches'][ind][:,j])(np.linspace(start, end, 1000))
        li_li_timbre[:,j] = Akima1DInterpolator(seg, subset['segments_timbre'][ind][:,j])(np.linspace(start, end, 1000))
    pitch.append(li_li_pitch)
    timbre.append(li_li_timbre)

np.save(o_dir+'/'+input_df[-8]+'pitch.npy', np.array(pitch))
np.save(o_dir+'/'+input_df[-8]+'timbre.npy', np.array(pitch))

# Processing 1D features next
# mean and std from StandardScaler acting on representative subset (A) 
mean = np.array([123.810, 3.600, 5.331, 0.671, -10.100])
std = np.sqrt([1230.218, 1.500, 12.816, 0.221, 26.985])
features1D = (subset[['tempo', 'time_signature', 'key', 'mode', 'loudness']] - mean)/std
np.save(o_dir+'/'+input_df[-8]+'features1D.npy', features1D)

# Processing labels last
red_terms, red_freq = [], []
for i in range(subset.shape[0]):
    mask = [True if x in terms80 else False for x in subset['artist_terms'][i]]
    red_terms.append(subset['artist_terms'][i][mask])
    red_freq.append(subset['artist_terms_freq'][i][mask])

labels = [{x:y for x,y in zip(A,B)} for A,B in zip(red_terms, red_freq)]

ohe = DictVectorizer(sparse=False)
labels = ohe.fit_transform(labels)

np.save(o_dir+'/'+input_df[-8]+'labels.npy', labels)

