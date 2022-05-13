# Utility script to analyze the results of the various models.
# Created by James Raphael Tiovalen (2022)

import json
import os
import pprint

import librosa
import matplotlib.pyplot as plt
import torch
import torchaudio

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

# Best model name
best_model = "ShallowCNN_lfcc_I"

# Selected data to add for dropdown in GUI showcase
selected_data = [
    "LJ049-0058.wav",
    "LJ043-0062.wav",
    "LJ042-0089.wav",
    "LJ049-0221.wav",
    "LJ042-0238.wav",
    "LJ045-0202.wav",
    "LJ048-0265.wav",
    "LJ049-0103.wav",
    "LJ046-0243.wav",
    "LJ049-0079.wav",
    "LJ047-0048.wav",
    "LJ043-0069.wav",
    "LJ048-0178.wav",
    "LJ044-0233.wav",
    "LJ042-0222.wav",
    "LJ040-0165.wav",
    "LJ040-0073.wav",
    "LJ045-0182.wav",
    "LJ045-0231.wav",
    "LJ050-0082.wav",
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
        else:
            axes[c].set_ylabel("Amplitude")
        axes[c].set_xlabel("Time (s)")
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

    # Print HTML tags to help build the GUI
    print("GENERATED HTML OPTION TAGS:\n")
    for filename in selected_data:
        print(f'<option value="{filename[:-4]}">{filename[:-4]}</option>')
    for filename in selected_data:
        print(f'<option value="{filename[:-4]}_gen">{filename[:-4]}_gen</option>')

    print("")

    # Build dictionary
    data_filename_dict = {
        data_filename.strip(): idx for idx, data_filename in enumerate(data_filenames)
    }

    selected_data_indices = []

    for data_filename in selected_data:
        filename = data_filename.strip()
        selected_data_indices.append((filename, data_filename_dict[filename]))
    for data_filename in selected_data:
        filename = data_filename.strip()[:-4] + "_gen.wav"
        selected_data_indices.append((filename, data_filename_dict[filename]))

    # Get cached inference from best model
    with open(f"saved/{best_model}/best_pred.json") as inference_file:
        model_data = json.load(inference_file)

    model_data_lst = list(zip(model_data["y_true"], model_data["y_pred"]))

    cached_inference = {}
    no_mispredictions = True

    for filename, data_idx in selected_data_indices:
        pruned_filename = filename[:-4]
        cached_inference[pruned_filename] = str(model_data_lst[data_idx][1])
        if model_data_lst[data_idx][0] != model_data_lst[data_idx][1]:
            print(
                f"{pruned_filename}'s ground truth label is not the same as its predicted label."
            )
            no_mispredictions = False

    if not no_mispredictions:
        print()
    print("CACHED INFERENCE:\n")
    pp = pprint.PrettyPrinter(depth=4, sort_dicts=False)
    pp.pprint(cached_inference)

    # By interesting, we mean that some of the models wrongly classified the data point
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

    anomaly_directory = "samples/anomalies"

    for file in os.listdir(anomaly_directory):
        f = os.path.join(anomaly_directory, file)
        if os.path.isfile(f):
            plot_actual_waveform(f)
            plot_actual_spectrogram(f)
            plot_mfcc(f)
            plot_lfcc(f)
