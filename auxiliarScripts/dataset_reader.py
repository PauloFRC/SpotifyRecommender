import json

class SpotifyPlaylistDataset:
    def __init__(self, verbose=True):
        self.verbose = verbose
        self.playlists = None
        self.current_slice = 0
        self._load_slice(0)
        
    def _load_slice(self, data_slice):
        if self.verbose:
            print(f"Carregando dados da slice {data_slice*1000}-{data_slice*1000+999}")
        with open(f'../data/mpd.slice.{data_slice*1000}-{data_slice*1000+999}.json') as f:
            self.playlists = json.load(f)['playlists']
        self.current_slice = data_slice
    
    def read_playlist(self, index):
        if not (self.current_slice*1000 <= index <= self.current_slice*1000+999):
            self._load_slice(index//1000)
        return self.playlists[index % 1000]