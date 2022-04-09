from torch.utils.data import Dataset
import torch
from utils import get_waveform, trim_or_pad_time
from utils import find_wav_files
from tqdm import tqdm
class WavDataset(Dataset):
    def __init__(self, wav_files, label, sample_rate = 16_000, pad_time=64600) -> None:
        """
        label == 1: real
        label == 0: fake
        """
        super().__init__()
        self.label = label
        self.sample_rate = sample_rate
        self.data = []
        data_type  = "real" if label == 1 else "fake"
        for file in tqdm(wav_files, desc="loading " + data_type + " data"):
            wav, rate = get_waveform(file, self.sample_rate)
            padded_wav = trim_or_pad_time(wav, 64600)
            self.data.append([padded_wav, rate])
            
        
    def __getitem__(self, index):
        """
        Returns:
            [waveform, samplerate, label]
        """
        return self.data[index] + [self.label]
    
    def __len__(self):
        return len(self.data)
    
    
def get_dataset(fake_dirs, real_dir,  train_val_test_split, debug=False):
    fake_files = [find_wav_files(fake_dir) for fake_dir in fake_dirs]
    fake_files = [ fake_file for ls in fake_files for fake_file in ls]
    real_files = find_wav_files(real_dir)
    
    if debug:
        fake_files =  fake_files[:1000]
        real_files =  real_files[:1000]
    fake_dataset = WavDataset(fake_files, 0)
    real_dataset = WavDataset(real_files, 1)
    dataset =  torch.utils.data.ConcatDataset([fake_dataset, real_dataset])
    
    train_len = int(len(dataset) *train_val_test_split[0])
    val_len = int(len(dataset) * train_val_test_split[1])
    test_len = int(len(dataset) * train_val_test_split[2])
    lengths = [train_len, val_len, test_len]
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, lengths)
    return train_dataset, val_dataset, test_dataset
    

