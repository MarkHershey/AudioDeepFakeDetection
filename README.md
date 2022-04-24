# Audio Deep Fake Detection

A Course Project for SUTD 50.039 Theory and Practice of Deep Learning (2022 Spring)

Created by [Mark He Huang](https://markhh.com/), [Peiyuan Zhang](https://www.linkedin.com/in/lance-peiyuan-zhang-5b2886194/), [James Raphael Tiovalen](https://jamestiotio.github.io/), [Madhumitha Balaji](https://www.linkedin.com/in/madhu-balaji/), and [Shyam Sridhar](https://www.linkedin.com/in/shyam-sridhar/).

Check out our: [Project Report](Report.pdf) | [GUI (Website)](https://markhh.com/AudioDeepFakeDetection/)

## Setup Environment

```bash
# Set up Python virtual environment
python3 -m venv venv && source venv/bin/activate

# Make sure your PIP is up to date
pip install -U pip wheel setuptools

# Install required dependencies
pip install -r requirements.txt
```

-   Install PyTorch that suits your machine: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

## Setup Datasets

You may download the datasets used in the project from the following URLs:

-   (Real) Human Voice Dataset: [LJ Speech (v1.1)](https://keithito.com/LJ-Speech-Dataset/)
    -   This dataset consists of 13,100 short audio clips of a single speaker reading passages from 7 non-fiction books.
-   (Fake) Synthetic Voice Dataset: [WaveFake (v1.20)](https://zenodo.org/record/5642694)
    -   The dataset consists of 104,885 generated audio clips (16-bit PCM wav).

After downloading the datasets, you may extract them under `data/real` and `data/fake` respectively. In the end, the `data` directory should look like this:

```
data
├── real
│   └── wavs
└── fake
    ├── common_voices_prompts_from_conformer_fastspeech2_pwg_ljspeech
    ├── jsut_multi_band_melgan
    ├── jsut_parallel_wavegan
    ├── ljspeech_full_band_melgan
    ├── ljspeech_hifiGAN
    ├── ljspeech_melgan
    ├── ljspeech_melgan_large
    ├── ljspeech_multi_band_melgan
    ├── ljspeech_parallel_wavegan
    └── ljspeech_waveglow
```

## Model Checkpoints

Model checkpoints are included in this repository using [Git Large File Storage (LFS)](https://git-lfs.github.com/).

If you already have `git-lfs` installed, the model checkpoints will be automatically downloaded when you run command `git clone` or `git pull`.

If you don't have `git-lfs` installed, you can follow the install instructions [here](https://git-lfs.github.com/). For debain-based linux users, you can install `git-lfs` using the following command:

```bash
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
sudo apt-get install git-lfs
```

After you have `git-lfs` installed, you can execute the following command at the root of this repository to download the model checkpoints:

```bash
git lfs pull
```

## Training

Use the [`train.py`](train.py) script to train the model.

```
usage: train.py [-h] [--real_dir REAL_DIR] [--fake_dir FAKE_DIR] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--seed SEED] [--feature_classname {wave,lfcc,mfcc}]
                [--model_classname {MLP,WaveRNN,WaveLSTM,SimpleLSTM,ShallowCNN,TSSD}]
                [--in_distribution {True,False}] [--device DEVICE] [--deterministic] [--restore] [--debug]
                [--debug_all]

optional arguments:
  -h, --help            show this help message and exit
  --real_dir REAL_DIR, --real REAL_DIR
                        Directory containing real data. (default: 'data/real')
  --fake_dir FAKE_DIR, --fake FAKE_DIR
                        Directory containing fake data. (default: 'data/fake')
  --batch_size BATCH_SIZE
                        Batch size. (default: 256)
  --epochs EPOCHS       Number of maximum epochs to train. (default: 20)
  --seed SEED           Random seed. (default: 42)
  --feature_classname {wave,lfcc,mfcc}
                        Feature classname. (default: 'lfcc')
  --model_classname {MLP,WaveRNN,WaveLSTM,SimpleLSTM,ShallowCNN,TSSD}
                        Model classname. (default: 'ShallowCNN')
  --in_distribution {True,False}, --in_dist {True,False}
                        Whether to use in distribution experiment setup. (default: True)
  --device DEVICE       Device to use. (default: 'cuda' if possible)
  --deterministic       Whether to use deterministic training (reproducible results).
  --restore             Whether to restore from checkpoint.
  --debug               Whether to use debug mode.
  --debug_all           Whether to use debug mode for all models.
```

Example:

To make sure all models can run successfully on your device, you can run the following command to test:

```bash
python train.py --debug_all
```

To train the model `ShallowCNN` with `lfcc` features in the in-distribution setting, you can run the following command:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --real data/real --fake data/fake --batch_size 128 --epochs 20 --seed 42 --feature_classname lfcc --model_classname ShallowCNN
```

## Evaluation Results

By default, we directly use test set for training validation, and the best model and the best predictions will be automatically saved in the [`saved`](saved) directory during training. Go to the directory [`saved`](saved) to see the evaluation results.

Run the following command to re-compute the evaluation results based on saved predictions and labels:

```bash
python metrics.py
```

## Acknowledgements

-   We thank [Dr. Matthieu De Mari](https://istd.sutd.edu.sg/people/faculty/matthieu-de-mari) and [Prof. Berrak Sisman](https://istd.sutd.edu.sg/people/faculty/berrak-sisman) for their teaching and guidance.
-   We thank Joel Frank and Lea Schönherr. Our code is partially adopted from their repository [WaveFake](https://github.com/RUB-SysSec/WaveFake).
-   We thank [Prof. Liu Jun](https://istd.sutd.edu.sg/people/faculty/liu-jun) for providing GPU resources for conducting experiments for this project.

## License

Our project is licensed under the [MIT License](LICENSE).
