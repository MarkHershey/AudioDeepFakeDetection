# AudioDeepFakeDetection

## Setup Environment

-   Step 1:
    ```bash
    # set up python virtual environment
    python3 -m venv venv && source venv/bin/activate
    # make sure your pip is up to date
    pip install -U pip wheel setuptools
    # install required dependencies
    pip install -r requirements.txt
    ```
-   Step 2:
    -   Install pytorch that suits your machine: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

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

## Evaluation

## Acknowledgements

-   Our code is partially adopted from [WaveFake](https://github.com/RUB-SysSec/WaveFake).
