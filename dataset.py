from torch.utils.data import Dataset
import torch
from utils import get_waveform, trim_or_pad_time
from utils import find_wav_files
from tqdm import tqdm
import random
class WavDataset(Dataset):
    def __init__(self, wav_files, label, sample_rate = 16_000, pad_time=64600) -> None:
        """
        label == 1: real
        label == 0: fake
        """
        super().__init__()
        self.label = label
        self.sample_rate = sample_rate
        self.data = {}
        self.wav_files = wav_files
        data_type  = "real" if label == 1 else "fake"
    
        
    def __getitem__(self, index):
        """
        Returns:
            [waveform, samplerate, label]
        """
        if index not in self.data:
            file = self.wav_files[index]
            wav, rate = get_waveform(file, self.sample_rate)
            padded_wav = trim_or_pad_time(wav, 64600)
            self.data[index] = [padded_wav, rate]
        return self.data[index] + [self.label]
    
    def __len__(self):
        return len(self.wav_files)
    
    
def get_dataset(fake_dirs, real_dir,  train_val_test_split, debug=False, down_sample_rate=1.0, experiment_1=False, experiment_2=False):
    assert experiment_1 or experiment_2
    fake_files = [find_wav_files(fake_dir) for fake_dir in fake_dirs]
    fake_files = [ fake_file for ls in fake_files for fake_file in ls]
    real_files = find_wav_files(real_dir)

    if down_sample_rate < 1.0:
        fake_files = random.sample(fake_files, int(len(fake_files)*down_sample_rate))
        real_files = random.sample(real_files, int(len(real_files)*down_sample_rate))
    if debug:
        fake_files =  fake_files[:1000]
        real_files =  real_files[:1000]
    fake_dataset = WavDataset(fake_files, 0)
    real_dataset = WavDataset(real_files, 1)
    sampler = None
    if experiment_1:
        train_dataset_fake_len = int(len(fake_dataset) * 0.8)
        train_dataset_fake, test_dataset_fake = torch.utils.data.random_split(fake_dataset, [train_dataset_fake_len,len(fake_dataset) -  train_dataset_fake_len])
        train_dataset_real_len  = int(len(real_dataset) * 0.8)
        train_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_dataset_real_len,len(real_dataset) -  train_dataset_real_len])
    if experiment_2:
        train_dataset_fake = fake_dataset
        test_dataset_fake_file = find_wav_files("../generated_audio/ljspeech_melgan")
        test_dataset_fake_file = random.sample(test_dataset_fake_file, int(len(test_dataset_fake_file)*0.2))
        test_dataset_fake = WavDataset(test_dataset_fake_file, 0)
        
        train_dataset_real_len  = int(len(real_dataset) * 0.8)
        train_dataset_real, test_dataset_real = torch.utils.data.random_split(real_dataset, [train_dataset_real_len,len(real_dataset) -  train_dataset_real_len])

    if experiment_2:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_fake]+ [train_dataset_real]*6)
    else:
        train_dataset = torch.utils.data.ConcatDataset([train_dataset_fake, train_dataset_real])
    test_dataset = torch.utils.data.ConcatDataset([test_dataset_fake, test_dataset_real])
    
    return train_dataset, test_dataset
    

