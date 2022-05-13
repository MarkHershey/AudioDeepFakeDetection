# Empirical Results
 
-   Accuracy
-   F1 score
-   Area Under the Receiver Operating Characteristic Curve (ROC AUC)
-   Equal Error Rate (EER)
 
| Experiment | Accuracy | F1 Score | ROC AUC | EER | EER2 |
| :--------- | :------: | :------: | :-----: | :-: | :--: |
| WaveLSTM_wave_O | 0.500 | 0.000 | 0.5000 | 0.5000 | 0.0000 |
| MLP_mfcc_I | 0.500 | 0.000 | 0.5000 | 0.5000 | 0.0000 |
| WaveRNN_wave_O | 0.500 | 0.000 | 0.5000 | 0.5000 | 0.0000 |
| WaveRNN_wave_I | 0.653 | 0.649 | 0.6527 | 0.3502 | 0.3378 |
| WaveLSTM_wave_I | 0.749 | 0.742 | 0.7494 | 0.2640 | 0.2221 |
| ShallowCNN_lfcc_O | 0.937 | 0.939 | 0.9366 | 0.0926 | 0.0992 |
| TSSD_wave_O | 0.956 | 0.957 | 0.9561 | 0.0561 | 0.0576 |
| SimpleLSTM_mfcc_I | 0.960 | 0.960 | 0.9601 | 0.0404 | 0.0405 |
| SimpleLSTM_lfcc_O | 0.965 | 0.965 | 0.9651 | 0.0441 | 0.0248 |
| SimpleLSTM_lfcc_I | 0.996 | 0.996 | 0.9962 | 0.0042 | 0.0034 |
| ShallowCNN_mfcc_I | 0.997 | 0.997 | 0.9968 | 0.0049 | 0.0015 |
| TSSD_wave_I | 0.999 | 0.999 | 0.9994 | 0.0011 | 0.0000 |
| ShallowCNN_lfcc_I | 1.000 | 1.000 | 0.9996 | 0.0004 | 0.0004 |