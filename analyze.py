# Utility script to analyze the results of the various models.
# Created by James Raphael Tiovalen (2021)

import json

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
            interesting_data_points_results.append(
                (dataset_idx, [models[i] for i in wrong_model_indexes])
            )

    print(interesting_data_points_results)
