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
    mainFolder = cwd + os.sep + 'all_xmls' + os.sep
    styles_folders = ['han' + os.sep, 'jazz' + os.sep]
    session_names = ['han', 'jazz']
    blending_sessions = [['han0120','fried_bananas'] , ['han0351','i_fall_in_love_too_easy'] , ['han0238','i_hear_rapsody'] , ['han0207','my_silient_love']]
    
    all_names = []
    all_features = []
    all_features_np = []
    blend_names = []
    for j in range(14):
        blend_names.append( 'blend_' + str(j) + '.xml' )
    blending_indexes = []
    
    # first construct the features matrix of all pieces in both styles
    for i in range( len( styles_folders ) ):
        print('Processing initial: ', styles_folders[i])
        folderName = mainFolder + styles_folders[i]
        all_files = glob.glob(folderName + "*.xml")
        tmp_feats = []
        # for all pieces extract features and put them in respective np.arrays
        for j in range( len( all_files ) ):
            fileName = all_files[j]
            all_names.append( fileName.split(os.sep)[-1] )
            # print('Processing initial: ', fileName)
            p = m21.converter.parse( fileName )
            tmp_val = mff.get_features_of_stream( p )
            tmp_feats.append( tmp_val )
            all_features.append( tmp_val )
    # end for styles
    # for each blending sessions, append features
    for i in range( len( blending_sessions ) ):
        session_folder = 'bl'+str(i+1)+'_'+blending_sessions[i][1]+'_'+blending_sessions[i][0]+os.sep
        blending_indexes.append( range( len(all_features), len(all_features)+len(blend_names), 1 ) )
        print('Processing blend: ', session_folder)
        for j in range( len( blend_names ) ):
            # print('Processing blend: ', blend_names[j])
            fileName = cwd+os.sep+'full'+os.sep+session_folder+ blend_names[j]
            p = m21.converter.parse( fileName )
            tmp_val = mff.get_features_of_stream( p )
            tmp_feats.append( tmp_val )
            all_features.append( tmp_val )
    # do PCA to all features
    # PCA
    pca = PCA(n_components=2)
    all_features_np = np.vstack( all_features )
    # normalise
    x = all_features_np
    x_max = np.max(x, axis=0)
    x_min = np.min(x, axis=0)
    y = (x-x_min)/(x_max-x_min);
    # all_pca = pca.fit_transform( np.vstack( all_features_np ) )
    all_pca = pca.fit_transform( np.vstack( y ) )
    tmp_pca = pca.fit(np.vstack( y ))
    explained = tmp_pca.explained_variance_ratio_
    print('PCA explained variances: ', explained)
    print('PCA axes correlations:')
    for i in range(2):
        for j in range(4):
            print('PCA_', i, ' - f_', j, ': ', np.corrcoef( all_pca[:,i], all_features_np[:,j] )[0][1])
    print('PCA_0 - f_0+f_2: ', np.corrcoef( all_pca[:,0], all_features_np[:,0]+all_features_np[:,2] )[0][1])
    # keep the pca coordinates of the original (not blended) pieces
    all_original_pca = all_pca[ :len(all_names) , : ]
    for i in range( len( blending_sessions ) ):
        # keep indexes of the pieces to be highlighted
        han2show = blending_sessions[i][0]
        jazz2show = blending_sessions[i][1]
        # get indexes to highlight
        han_idx = all_names.index( han2show+'.xml' )
        jazz_idx = all_names.index( jazz2show+'.xml' )
        # get pca of blends on features matrix
        blends_pca = all_pca[ blending_indexes[i] , : ]
        # plot pca of original pieces
        plt.plot(all_original_pca[:488,0], all_original_pca[:488,1], '|', color='grey', alpha=0.5, label='Han')
        plt.plot(all_original_pca[488:,0], all_original_pca[488:,1], '_', color='grey', alpha=0.5, label='Jazz')
        # highlight inputs
        plt.plot(all_original_pca[han_idx,0], all_original_pca[han_idx,1], 'd', color='grey', markerSize=10, label='Han input')
        plt.plot(all_original_pca[jazz_idx,0], all_original_pca[jazz_idx,1], 's', color='grey', markerSize=10, label='Jazz input')
        # plot blends
        for j in range(14):
            if j==0:
                plt.plot(blends_pca[j,0], blends_pca[j,1], '.', color='black', label='blend')
            else:
                plt.plot(blends_pca[j,0], blends_pca[j,1], '.', color='black')
            plt.text(blends_pca[j,0], blends_pca[j,1], str(j), color='black', fontsize=9, bbox=dict(facecolor='white', alpha=0.2))
        plt.xticks([])
        plt.yticks([])
        plt.legend()
        plt.savefig('figs/pca_'+blending_sessions[i][1]+'_'+blending_sessions[i][0]+'.png', dpi=500); plt.clf()
    # save
    with open('saved_data/all_features.pickle', 'wb') as handle:
        pickle.dump(all_features, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/all_pca.pickle', 'wb') as handle:
        pickle.dump(all_pca, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open('saved_data/all_names.pickle', 'wb') as handle:
        pickle.dump(all_names, handle, protocol=pickle.HIGHEST_PROTOCOL)
else:
    with open('saved_data/all_features.pickle', 'rb') as handle:
        all_features = pickle.load(handle)
    with open('saved_data/all_pca.pickle', 'rb') as handle:
        all_pca = pickle.load(handle)
    with open('saved_data/all_names.pickle', 'rb') as handle:
        all_names = pickle.load(handle)
    # PCA
    pca = PCA(n_components=2)
    all_features_np = np.vstack( all_features )
    all_pca = pca.fit_transform( np.vstack( all_features_np ) )
    plt.plot(all_pca[:447,0], all_pca[:447,1], '|', color='grey', alpha=0.5)
    plt.plot(all_pca[447:,0], all_pca[447:,1], '_', color='grey', alpha=0.5)
    for i in range( len( han_highlight ) ):
        plt.plot(all_pca[han_highlight[i],0], all_pca[han_highlight[i],1], 'd', color='black')
        plt.plot(all_pca[jazz_highlight[i],0], all_pca[jazz_highlight[i],1], 's', color='black')
    plt.savefig('figs/pca_all.png', dpi=500); plt.clf()

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