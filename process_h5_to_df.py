import numpy as np
import pandas as pd
import sys
import re
import os
import gc
import h5py
import hdf5_getters
from optparse import OptionParser
from ediblepickle import checkpoint

parser = OptionParser()
parser.add_option('--i_dir',dest='i_dir',help='input directory',default='./')
parser.add_option('--o_dir',dest='o_dir',help='output directory',default='./')

(options, args) = parser.parse_args()
i_dir = options.i_dir
o_dir = options.o_dir

cache_dir = 'h5_to_df_cache'
if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)

if not os.path.exists(o_dir):
    os.mkdir(o_dir)

@checkpoint(key=lambda args, kwargs: re.sub('/|MillionSong','',args[0]) +'.pkl', work_dir=cache_dir)
def gen_df_h5(my_dir):
    '''
    Pulls information from all .h5 files and stores as a pandas DataFrame
    Target .h5 files are from Million Song Dataset
    Data being pulled, in order, includes
        song_ID, artist_ID, title, artist_name, release, artist_7digitalid, track_7digitalid, 
        analysis_sample_rate, sections_start, segments_pitches, segments_timbre, segments_start, 
        tempo, time_signature, key, mode, end_of_fade_in, start_of_fade_out, duration, danceability, 
        energy, loudness, song_hotttnesss, artist_hotttnesss, artist_mbtags, artist_terms, 
        artist_terms_freq, artist_terms_weight, similar_artists
    Inputs:
        my_dir: str
            directory containined .h5 files
    Outputs:
        df: pandas dataframe
            dataframe containing information from .h5 file
    '''
    columns = '''song_ID, artist_ID, title, artist_name, release, artist_7digitalid, track_7digitalid, 
        analysis_sample_rate, sections_start, segments_pitches, segments_timbre, segments_start, 
        tempo, time_signature, key, mode, end_of_fade_in, start_of_fade_out, duration, danceability, 
        energy, loudness, song_hotttnesss, artist_hotttnesss, artist_mbtags, artist_terms, 
        artist_terms_freq, artist_terms_weight, similar_artists'''
    columns = re.split(', ',re.sub('\n\s*','',columns))

    df = pd.DataFrame(columns=columns)
    files = os.listdir(my_dir)

    for f in files:
        h5 = hdf5_getters.open_h5_file_read(my_dir+f)
        my_dict = dict.fromkeys(columns)

        my_dict['song_ID'] = [hdf5_getters.get_song_id(h5).decode()]
        my_dict['artist_ID'] = [hdf5_getters.get_artist_id(h5).decode()]
        my_dict['title'] = [hdf5_getters.get_title(h5).decode()]
        my_dict['artist_name'] = [hdf5_getters.get_artist_name(h5).decode()]
        my_dict['release'] = [hdf5_getters.get_release(h5).decode()]
        my_dict['artist_7digitalid'] = [hdf5_getters.get_artist_7digitalid(h5)]
        my_dict['track_7digitalid'] = [hdf5_getters.get_track_7digitalid(h5)]
        my_dict['analysis_sample_rate'] = [hdf5_getters.get_analysis_sample_rate(h5)]
        my_dict['sections_start'] = [hdf5_getters.get_sections_start(h5)]
        my_dict['segments_pitches'] = [hdf5_getters.get_segments_pitches(h5)]
        my_dict['segments_timbre'] = [hdf5_getters.get_segments_timbre(h5)]
        my_dict['segments_start'] = [hdf5_getters.get_segments_start(h5)]
        my_dict['tempo']  = [hdf5_getters.get_tempo(h5)]
        my_dict['time_signature'] = [hdf5_getters.get_time_signature(h5)]
        my_dict['key'] = [hdf5_getters.get_key(h5)]
        my_dict['mode'] = [hdf5_getters.get_mode(h5)]
        my_dict['end_of_fade_in'] = [hdf5_getters.get_end_of_fade_in(h5)]
        my_dict['start_of_fade_out'] = [hdf5_getters.get_start_of_fade_out(h5)]
        my_dict['duration'] = [hdf5_getters.get_duration(h5)]
        my_dict['danceability'] = [hdf5_getters.get_danceability(h5)]
        my_dict['energy'] = [hdf5_getters.get_energy(h5)]
        my_dict['loudness'] = [hdf5_getters.get_loudness(h5)]
        my_dict['song_hotttnesss'] = [hdf5_getters.get_song_hotttnesss(h5)]
        my_dict['artist_hotttness'] = [hdf5_getters.get_artist_hotttnesss(h5)]
        my_dict['artist_mbtags'] = [np.array([x.decode() for x in hdf5_getters.get_artist_mbtags(h5)])]
        my_dict['artist_terms'] = [hdf5_getters.get_artist_terms(h5).astype(str)]
        my_dict['artist_terms_freq'] = [hdf5_getters.get_artist_terms_freq(h5)]
        my_dict['artist_terms_weight'] = [hdf5_getters.get_artist_terms_weight(h5)]
        my_dict['similar_artists'] = [hdf5_getters.get_similar_artists(h5).astype(str)]

        df = pd.concat([df, pd.DataFrame(my_dict)], axis=0, ignore_index=True)

    return df

def merge_df(my_dir, i_dir):
    '''
    Merges pickled df files
    Returns combined df
    '''
    li = []

    files = os.listdir(my_dir)
    for f in files:
        if re.sub('/','',i_dir) in f:
            df = pd.read_pickle(my_dir+'/'+f)
            for x in df.to_dict(orient='records'):
                li.append(x)

    return pd.DataFrame(li)

if re.sub('/','',i_dir)+'_df.pkl' in os.listdir(o_dir):
    sys.exit(i_dir+' already processed and can be found as '+re.sub('/','',i_dir)+'_df.pkl'+' in '+o_dir)

root = i_dir
for lett in os.listdir(root):
    for d in os.listdir(root+lett+'/'):
        gen_df_h5(root+lett+'/'+d+'/')
        gc.collect()

total_df = merge_df('h5_to_df_cache/', i_dir)
total_df.to_pickle(o_dir+'/'+re.sub('/','',i_dir)+'_df.pkl')

