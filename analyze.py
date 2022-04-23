# Utility script to analyze the results of the various models.
# Created by James Raphael Tiovalen (2022)

import json
import os

import librosa
import torch
import torchaudio
import matplotlib.pyplot as plt

from module.lfcc import LFCC

# We only consider and include the qualified models here
models = [
    "ShallowCNN_lfcc_I",
    "ShallowCNN_lfcc_O",
    "ShallowCNN_mfcc_I",
    "SimpleLSTM_lfcc_I",
    "SimpleLSTM_lfcc_O",
    "SimpleLSTM_mfcc_I",
    "TSSD_wave_I",
    "TSSD_wave_O",
    "WaveLSTM_wave_I",
    "WaveRNN_wave_I",
]


def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    figure.suptitle(title)
    plt.show()


def plot_spectrogram(spec, title=None, ylabel="freq_bin", aspect="auto", xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Spectrogram (dB)")
    axs.set_ylabel(ylabel)
    axs.set_xlabel("frame")
    im = axs.imshow(librosa.power_to_db(spec), origin="lower", aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs, format="%+2.0f dB")
    plt.show()


def plot_actual_waveform(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    plot_waveform(waveform, sample_rate, title=f"{audio_path} (Waveform)")


def plot_actual_spectrogram(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    spectrogram_transform = torchaudio.transforms.Spectrogram()
    spectrogram = spectrogram_transform(waveform)
    plot_spectrogram(spectrogram[0], title=f"{audio_path} (Spectrogram)")


def plot_mfcc(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate)
    mfcc = mfcc_transform(waveform)
    plot_spectrogram(mfcc[0], title=f"{audio_path} (MFCC)")


def plot_lfcc(audio_path):
    waveform, sample_rate = torchaudio.load(audio_path)
    lfcc_transform = LFCC(sample_rate=sample_rate)
    lfcc = lfcc_transform(waveform)
    plot_spectrogram(lfcc[0], title=f"{audio_path} (LFCC)")


if __name__ == "__main__":
    with open("saved/testing_audio_names.txt") as data_filename_file:
        data_filenames = data_filename_file.readlines()

    # By interesting, we mean that some of the models wrongly classified the data point.
    interesting_data_points_results = []
    for dataset_idx in range(5240):
        single_data_point_result = []
        for model in models:
            with open("saved/" + str(model) + "/best_pred.json") as json_file:
                data = json.load(json_file)
                single_data_point_result.append(
                    (data["y_true"][dataset_idx], data["y_pred"][dataset_idx])
                )

        # Get the datasets that 5 models predicted wrongly on
        # This is the worst that we can get so far
        if sum(i != j for i, j in single_data_point_result) == 5:
            wrong_model_indexes = [
                idx for idx, (i, j) in enumerate(single_data_point_result) if i != j
            ]
            data_point = data_filenames[dataset_idx].strip()
            interesting_data_points_results.append(
                (data_point, [models[i] for i in wrong_model_indexes])
            )

    print(interesting_data_points_results)

    anomaly_directory = "anomalies"

    for file in os.listdir(anomaly_directory):
        f = os.path.join(anomaly_directory, file)
        if os.path.isfile(f):
            plot_actual_waveform(f)
            plot_actual_spectrogram(f)
            plot_mfcc(f)
            plot_lfcc(f)
