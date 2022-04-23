# AudioDeepFakeDetection

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

If you already have `git-lfs` installed, the model checkpoints will be automatically downloaded when command running `git clone` or `git pull`.

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
                [--in_distribution IN_DISTRIBUTION] [--device DEVICE] [--deterministic] [--restore] [--debug]
                [--debug_all]

optional arguments:
  -h, --help            show this help message and exit
  --real_dir REAL_DIR, --real REAL_DIR
                        Directory containing real data.
  --fake_dir FAKE_DIR, --fake FAKE_DIR
                        Directory containing fake data.
  --batch_size BATCH_SIZE
                        Batch size.
  --epochs EPOCHS       Number of maximum epochs to train.
  --seed SEED           Random seed.
  --feature_classname {wave,lfcc,mfcc}
                        Feature classname. One of: wave, lfcc, mfcc
  --model_classname {MLP,WaveRNN,WaveLSTM,SimpleLSTM,ShallowCNN,TSSD}
                        Model classname. One of: MLP, WaveRNN, WaveLSTM, SimpleLSTM, ShallowCNN, TSSD
  --in_distribution IN_DISTRIBUTION, --in_dist IN_DISTRIBUTION
                        Whether to use in distribution experiment setup.
  --device DEVICE       Device to use.
  --deterministic       Whether to use deterministic training (fix random seed).
  --restore             Whether to restore from checkpoint.
  --debug               Whether to use debug mode.
  --debug_all           Whether to use debug mode for all models.
```

Example:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py --real data/real --fake data/fake --batch_size 128 --epochs 20 --seed 42 --feature_classname lfcc --model_classname ShallowCNN

```

## Evaluation Results

Go to the directory [saved](saved) to see the evaluation results.

Run the following command to re-compute the evaluation results based on saved predictions and labels:

```bash
python metrics.py
```

## Acknowledgements

-   Our code is partially adopted from [WaveFake](https://github.com/RUB-SysSec/WaveFake).

## License

Our project is licensed under the [MIT License](LICENSE).
