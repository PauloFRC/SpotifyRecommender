from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from scipy.spatial.distance import cdist
from IPython.display import display
from sklearn.metrics import davies_bouldin_score, silhouette_score
import sys
sys.path.append('../auxiliarScripts')
from dataset_reader import SpotifyPlaylistDataset
from yellowbrick.cluster import KElbowVisualizer

tracks = pd.read_csv('../tracks/tracks.csv', index_col='id')

music_ids = tracks[['name', 'artists']]
# remove colunas que não serão utilizadas
tracks.drop(['name', 'artists', 'explicit', 'time_signature', 'mode', 'key', 'duration_ms', 'popularity'], axis=1, inplace=True)

def scaler(X, min_=None, max_=None):
    if not min_:
        min_ = X.min()
    if not max_:
        max_ = X.max()
    return X.map(lambda x: (x - min_) / (max_ - min_))

tracks['loudness'] = scaler(tracks['loudness'], min_=-60, max_=0)
tracks['tempo'] = scaler(tracks['tempo'])

class Similarity:
    def __init__(self, distance='euclidean'):
        self.distance = distance
        self.tracks = None
        
    def fit(self, tracks, id_tracks):
        self.tracks = tracks
        self.id_tracks = id_tracks
    
    def recommend(self, user_tracks, n_tracks=5):
        # cria a música média
        mean_track = np.mean(user_tracks, axis=0)
        
        # calcula distancias e pega os indices das k músicas com menores distâncias
        dists = cdist([mean_track], self.tracks, self.distance)[0]
        k_arg_dists = np.argpartition(dists, n_tracks+1)[:n_tracks+1]
        
        k_ids = self.id_tracks.iloc[k_arg_dists]
        k_info = self.tracks.iloc[k_arg_dists]
        k_dists = pd.DataFrame(dists[k_arg_dists], columns=['Distâncias'], index=self.tracks.index[k_arg_dists])
        
        # retorna um dataframe com informação do nome, artistas, distância e features ordenadas pela distância
        return pd.concat([k_dists, k_ids, k_info], axis=1).sort_values(by=['Distâncias'])

class SimilarityMeans:
    def __init__(self, distance='euclidean'):
        self.distance = distance
        self.tracks = None
        self.id_tracks = None
        
    def fit(self, tracks, id_tracks):
        self.tracks = tracks
        self.id_tracks = id_tracks
        
    def clusterfy(self, user_tracks):
        km = KMeans()
        visualizer = KElbowVisualizer(km, k=(1,len(user_tracks)+1))
        visualizer.fit(user_tracks)
        
        elbow_v = visualizer.elbow_value_
        new_km = KMeans(n_clusters=elbow_v)
        new_km.fit(user_tracks)
        
        return new_km
    
    def recommend_clusters(self, user_tracks, n_tracks=5, playlist_in_playlists=True):
        if playlist_in_playlists:
            self.tracks = self.tracks.drop(index=user_tracks.index)
            self.id_tracks = self.id_tracks.drop(index=user_tracks.index)
        
        if len(user_tracks)==1:
            return [self.recommend(user_tracks[0], n_tracks)]
        
        model = self.clusterfy(user_tracks)
        clusters = model.cluster_centers_
        tracks = []
        for i in range(len(clusters)) :
            rec = self.recommend(clusters[i], n_tracks)
            tracks.append(rec)
        
        return tracks, pd.Series(model.labels_), clusters
        
    def recommend(self, user_track, n_tracks=5):
        # calcula distancias e pega os indices das k músicas com menores distâncias
        dists = cdist([user_track], self.tracks, self.distance)[0]
        k_arg_dists = np.argpartition(dists, n_tracks)[:n_tracks]
        
        k_ids = self.id_tracks.iloc[k_arg_dists]
        k_info = self.tracks.iloc[k_arg_dists]
        k_dists = pd.DataFrame(dists[k_arg_dists], columns=['Distâncias'], index=self.tracks.index[k_arg_dists])
        
        # retorna um dataframe com informação do nome, artistas, distância e features ordenadas pela distância
        return pd.concat([k_dists, k_ids, k_info], axis=1).sort_values(by=['Distâncias'])