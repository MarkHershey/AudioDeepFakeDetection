# Utility script to analyze the results of the various models.
# Created by James Raphael Tiovalen (2022)

import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt

models = [
    "MLP_mfcc_I",
    "ShallowCNN_lfcc_I",
    "ShallowCNN_lfcc_O",
    "ShallowCNN_mfcc_I",
    "SimpleLSTM_lfcc_I",
    "SimpleLSTM_lfcc_O",
    "SimpleLSTM_mfcc_I",
    "TSSD_wave_I",
    "TSSD_wave_O",
    "WaveLSTM_wave_I",
]

if __name__ == "__main__":
    with open("testing_audio_names.txt") as data_filename_file:
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
            # Plot MFCC feature
            audio_path = f
            fig, ax = plt.subplots()
            x, sr = librosa.load(audio_path)
            mfccs = librosa.feature.mfcc(x, sr=sr)
            img = librosa.display.specshow(mfccs, sr=sr, x_axis="time", ax=ax)
            ax.set(title=f"{f} (MFCC)")
            fig.colorbar(img, ax=ax)
            plt.show()
