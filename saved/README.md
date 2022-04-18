# Empirical Results
 
- Accuracy
- F1 score
- Area Under the Receiver Operating Characteristic Curve (ROC AUC)
- Equal Error Rate (EER)
 
| Experiment | Accuracy | F1 | ROC AUC | EER | EER2 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| MLP_mfcc_I | 0.500 | 0.000 | 0.5000 | 0.5000 | 0.0000 |
| WaveLSTM_wave_I | 0.754 | 0.746 | 0.7536 | 0.2600 | 0.2179 |
| TSSD_wave_O | 0.899 | 0.897 | 0.8989 | 0.1164 | 0.0813 |
| ShallowCNN_lfcc_O | 0.930 | 0.927 | 0.9296 | 0.0970 | 0.0374 |
| SimpleLSTM_mfcc_I | 0.960 | 0.960 | 0.9601 | 0.0404 | 0.0405 |
| SimpleLSTM_lfcc_O | 0.965 | 0.965 | 0.9651 | 0.0441 | 0.0248 |
| SimpleLSTM_lfcc_I | 0.996 | 0.996 | 0.9962 | 0.0042 | 0.0034 |
| ShallowCNN_mfcc_I | 0.997 | 0.997 | 0.9968 | 0.0049 | 0.0015 |
| TSSD_wave_I | 0.999 | 0.999 | 0.9990 | 0.0011 | 0.0011 |
| ShallowCNN_lfcc_I | 1.000 | 1.000 | 0.9996 | 0.0004 | 0.0004 |