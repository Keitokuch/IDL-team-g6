import torch
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
import numpy as np
from utils import *
from data_augment import *

class KnnwSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 audio_data,
                 subtitle_df,
                 data_aug=False,
                 total_frames=1370582, 
                 total_duration=6396010):
        self.duration_per_frame = total_duration / total_frames
        self.audio = audio_data
        self.subtitle_df = subtitle_df
        self.length = len(self.subtitle_df)
        self.data_aug = data_aug
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        start_time = self.subtitle_df.iloc[i]['Start time in ms']
        stop_time = self.subtitle_df.iloc[i]['End time in ms']
        
        audio_range = self.get_range(start_time, stop_time)
        
        audio_item = self.audio[audio_range, :]
        audio_item = torch.tensor(audio_item.astype(np.float32))
        
        subtitle_item = self.subtitle_df.iloc[i]['Transcript Indices']
        subtitle_item = torch.tensor(subtitle_item)

        if self.data_aug:
            audio_item = specaugment(audio_item)
        
        return audio_item, subtitle_item
        
    def get_index(self, time, start_flag):
        if start_flag == True:
            return np.floor(time/self.duration_per_frame)
        else:
            return np.ceil(time/self.duration_per_frame)
        
    def get_range(self, start_time, end_time):
        start_index = self.get_index(start_time, start_flag=True)
        stop_index  = self.get_index(end_time, start_flag=False)
        
        return range(int(start_index), int(stop_index))

    @classmethod
    def collate(cls, data):
        if len(data[0]) == 2:
            X, Y = zip(*data)
            y_lens = torch.tensor(list(map(len, Y)))
            x_lens = torch.tensor(list(map(len, X)))
            X = rnn.pad_sequence(X, batch_first=True)
            Y = rnn.pad_sequence(Y, batch_first=True)
            return X, Y, x_lens, y_lens
        else:
            X = data
            x_lens = torch.tensor(list(map(len, X)))
            X = rnn.pad_sequence(X, batch_first=True)
            return X, x_lens


class KnnwSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 audio_data,
                 subtitle_df,
                 total_frames=1370582, 
                 total_duration=6396010):
        self.duration_per_frame = total_duration / total_frames
        self.audio = audio_data
        self.subtitle_df = subtitle_df
        self.length = len(self.subtitle_df)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        start_time = self.subtitle_df.iloc[i]['Start time in ms']
        stop_time = self.subtitle_df.iloc[i]['End time in ms']
        
        audio_range = self.get_range(start_time, stop_time)
        
        audio_item = self.audio[audio_range, :]
        # audio_item = torch.tensor(audio_item.astype(np.float32))
        
        label_item = label2index[self.subtitle_df.iloc[i]['Speaker Label']]
        # label = torch.tensor(subtitle_item)
        
        return audio_item, label_item
        
    def get_index(self, time, start_flag):
        if start_flag == True:
            return np.floor(time/self.duration_per_frame)
        else:
            return np.ceil(time/self.duration_per_frame)
        
    def get_range(self, start_time, end_time):
        start_index = self.get_index(start_time, start_flag=True)
        stop_index  = self.get_index(end_time, start_flag=False)
        
        return range(int(start_index), int(stop_index))

    @classmethod
    def collate(cls, data):
        if len(data[0]) == 2:
            X, Y = zip(*data)
            X = [torch.tensor(x).float() for x, y in data]
            x_lens = torch.tensor(list(map(len, X)))
            X = rnn.pad_sequence(X, batch_first=True)
            return X, x_lens, torch.tensor(Y, dtype=torch.long)
        else:
            X = data
            X = [torch.tensor(x).float() for x, y in data]
            x_lens = torch.tensor(list(map(len, X)))
            X = rnn.pad_sequence(X, batch_first=True)
            return X, x_lens


class KnnwDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 audio_data,
                 subtitle_df,
                 total_frames=1370582, 
                 total_duration=6396010):
        self.duration_per_frame = total_duration / total_frames
        self.audio = audio_data
        self.subtitle_df = subtitle_df
        self.length = len(self.subtitle_df)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, i):
        start_time = self.subtitle_df.iloc[i]['Start time in ms']
        stop_time = self.subtitle_df.iloc[i]['End time in ms']
        
        audio_range = self.get_range(start_time, stop_time)
        
        audio_item = self.audio[audio_range, :]
        audio_item = torch.tensor(audio_item.astype(np.float32))
        
        label_item = label2index[self.subtitle_df.iloc[i]['Speaker Label']]
        subtitle_item = self.subtitle_df.iloc[i]['Transcript Indices']
        subtitle_item = torch.tensor(subtitle_item)
        
        return audio_item, subtitle_item, label_item
        
    def get_index(self, time, start_flag):
        if start_flag == True:
            return np.floor(time/self.duration_per_frame)
        else:
            return np.ceil(time/self.duration_per_frame)
        
    def get_range(self, start_time, end_time):
        start_index = self.get_index(start_time, start_flag=True)
        stop_index  = self.get_index(end_time, start_flag=False)
        
        return range(int(start_index), int(stop_index))

    @staticmethod
    def collate(data):
        if len(data[0]) == 3:
            X, Y, labs = zip(*data)
            x_lens = torch.tensor(list(map(len, X)))
            y_lens = torch.tensor(list(map(len, Y)))
            X = rnn.pad_sequence(X, batch_first=True)
            Y = rnn.pad_sequence(Y, batch_first=True)
            labs = torch.tensor(labs, dtype=torch.long)
            return X, Y, x_lens, y_lens, labs
        else:
            X = data
            x_lens = torch.tensor(list(map(len, X)))
            X = rnn.pad_sequence(X, batch_first=True)
            return X, x_lens
