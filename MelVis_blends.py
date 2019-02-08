#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 06:43:32 2018

@author: maximoskaliakatsos-papakostas
"""

import os
cwd = os.getcwd()
import glob
import music21 as m21
import numpy as np
from sklearn.decomposition import PCA
import MBL_melody_features_functions as mff
import CM_user_output_functions as uof
import MBL_music_processing_functions as mpf
import pickle
import matplotlib.pyplot as plt

remakedata = True
test_plot = True

if remakedata:
    mainFolder = cwd + '/full/'
    session_folders = ['bl1_benny_and_the_jets_han0269/', 'bl2_blues_in_the_night_han0284/', 'bl3_jersy_bounce_han0163/', 'bl4_please_dont_talk_about_me_han01/']
    session_names = ['b1', 'b2', 'b3', 'b4']
    
    all_names = []
    all_features = []
    all_features_np = []
    
    for i in range( len( session_folders ) ):
        session_features = []
        folderName = mainFolder + session_folders[i]
        all_files = ['han_input.mid', 'jazz_input.mid']
        for j in range(14):
            all_files.append( 'blend_' + str(j) + '.xml' )
        tmp_feats = []
        # for all pieces extract features and put them in respective np.arrays
        for j in range( len( all_files ) ):
            fileName = all_files[j]
            all_names.append( fileName.split('/')[-1] )
            print('Processing: ', fileName)
            p = m21.converter.parse( folderName + fileName )
            tmp_val = mff.get_features_of_stream( p )
            tmp_feats.append( tmp_val )
            all_features.append( tmp_val )
            session_features.append( tmp_val )
        # PCA
        pca = PCA(n_components=2)
        session_features_np = np.vstack( session_features )
        all_pca = pca.fit_transform( np.vstack( session_features_np ) )
        plt.plot(all_pca[2:,0], all_pca[2:,1], '.', color='grey')
        plt.plot(all_pca[0,0], all_pca[0,1], 'x', color='black')
        plt.plot(all_pca[1,0], all_pca[1,1], 'o', color='black')
        plt.savefig('figs/pca_'+session_names[i]+'.png', dpi=500); plt.clf()
    # end for sessions

'''
    # PCA
    pca = PCA(n_components=2)
    all_features_np = np.vstack( all_features )
    all_pca = pca.fit_transform( np.vstack( all_features_np ) )
    
    # sort by distance to other centroid
    np_styles_idx = np.array( all_styles_idx )
    pca_1 = all_pca[ np_styles_idx == 0 , : ]
    pca_2 = all_pca[ np_styles_idx == 1 , : ]
    features_1 = all_features_np[ np_styles_idx == 0 , : ]
    features_2 = all_features_np[ np_styles_idx == 1 , : ]
    centr_1 = np.mean(pca_1, axis = 0)
    centr_2 = np.mean(pca_2, axis = 0)
    # distances
    x = np.linalg.norm(pca_1 - centr_2, axis=1)
    y = 1/(np.linalg.norm(pca_1 - centr_1, axis=1)+1)
    d_pca_1 = x/np.max(x) + 0.3*y/np.max(y)
    x = np.linalg.norm(pca_2 - centr_1, axis=1)
    y = 1/(np.linalg.norm(pca_2 - centr_2, axis=1)+1)
    d_pca_2 = x/np.max(x) + 0.3*y/np.max(y)
    # get indexes of sorted distances
    sidx1 = np.argsort( d_pca_1 )[::-1]
    sidx2 = np.argsort( d_pca_2 )[::-1]
    # keep names of each style
    idxs_1 = np.where( np_styles_idx == 0 )[0]
    names_1 = [all_names[i] for i in idxs_1]
    idxs_2 = np.where( np_styles_idx == 1 )[0]
    names_2 = [all_names[i] for i in idxs_2]
    # keep shorted names
    s_names_1 = [names_1[i] for i in sidx1]
    s_names_2 = [names_2[i] for i in sidx2]
    # keep sorted pcas
    s_pca_1 = pca_1[ sidx1, : ]
    s_pca_2 = pca_2[ sidx2, : ]
    s_features_1 = features_1[ sidx1, : ]
    s_features_2 = features_2[ sidx2, : ]
    
    with open('saved_data/all_names.pickle', 'wb') as handle:
        pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/all_styles.pickle', 'wb') as handle:
        pickle.dump(all_styles, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/all_styles_idx.pickle', 'wb') as handle:
        pickle.dump(all_styles_idx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/all_features.pickle', 'wb') as handle:
        pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/all_pca.pickle', 'wb') as handle:
        pickle.dump(all_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/style_folders.pickle', 'wb') as handle:
        pickle.dump(style_folders, handle, protocol=pickle.HIGHEST_PROTOCOL)
    # save sorted pcas and names
    with open('saved_data/s_pca_1.pickle', 'wb') as handle:
        pickle.dump(s_pca_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/s_pca_2.pickle', 'wb') as handle:
        pickle.dump(s_pca_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/s_features_1.pickle', 'wb') as handle:
        pickle.dump(s_features_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/s_features_2.pickle', 'wb') as handle:
        pickle.dump(s_features_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/s_names_1.pickle', 'wb') as handle:
        pickle.dump(s_names_1, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/s_names_2.pickle', 'wb') as handle:
        pickle.dump(s_names_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('saved_data/all_names.pickle', 'rb') as handle:
        all_names = pickle.load(handle)
    with open('saved_data/all_styles.pickle', 'rb') as handle:
        all_styles = pickle.load(handle)
    with open('saved_data/all_styles_idx.pickle', 'rb') as handle:
        all_styles_idx = pickle.load(handle)
    with open('saved_data/all_features.pickle', 'rb') as handle:
        all_features = pickle.load(handle)
    with open('saved_data/all_pca.pickle', 'rb') as handle:
        all_pca = pickle.load(handle)
    with open('saved_data/style_folders.pickle', 'rb') as handle:
        style_folders = pickle.load(handle)
    # load sorted pcas and names
    with open('saved_data/s_pca_1.pickle', 'rb') as handle:
        s_pca_1 = pickle.load(handle)
    with open('saved_data/s_pca_2.pickle', 'rb') as handle:
        s_pca_2 = pickle.load(handle)
    with open('saved_data/s_features_1.pickle', 'rb') as handle:
        s_features_1 = pickle.load(handle)
    with open('saved_data/s_features_2.pickle', 'rb') as handle:
        s_features_2 = pickle.load(handle)
    with open('saved_data/s_names_1.pickle', 'rb') as handle:
        s_names_1 = pickle.load(handle)
    with open('saved_data/s_names_2.pickle', 'rb') as handle:
        s_names_2 = pickle.load(handle)
# end if remakedata

if test_plot:
    how_many = 100
    # style 1
    hm = min( [how_many, s_pca_1.shape[0]] )
    plt.plot( s_pca_1[ :hm , 0 ], s_pca_1[ :hm , 1 ], 'o' , label=style_folders[0] )
    # style 2
    hm = min( [how_many, s_pca_2.shape[0]] )
    plt.plot( s_pca_2[ :hm , 0 ], s_pca_2[ :hm , 1 ], 'x' , label=style_folders[1] )
    plt.legend()
    plt.savefig('figs/pca.png', dpi=300); plt.clf()
'''