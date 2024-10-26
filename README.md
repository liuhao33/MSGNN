
# MSGNN

This is the official implementation of MSGNN for Availability Check.

# System Requirements

## Hardware Platform

This code has been tested on the following platform:

+ CPU: I9-14900K

+ GPU: RTX 4090

+ Memory: 64GB

## Software requirements

### OS requirements

This code has been tested on the following systems:

+ Windows 10 (with CUDA 11.6)

### Dependencies

+ python 3.7

+ pytorch 1.12.1

+ scipy 1.7.3

+ networkx 2.6.3

+ scikit-learn 1.0.2

+ dask 2021.10.0
  
+ ipykernel

+ matplotlib

+ nltk (not required, see [below](#train-and-test-model-on-dblp))

# How to use

## Install dependencies

```shell
conda env create -f env.yaml
conda activate MSGNN
```

or

```shell
conda create -n MSGNN python=3.7
conda activate MSGNN
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
conda install scipy=1.7.3 networkx=2.6.3 scikit-learn=1.0.2 dask=2021.10.0
conda install ipykernel matplotlib

```

## Train and test model on DBLP

The DBLP dataset is cleaned by [MAGNN](https://github.com/cynricfu/MAGNN), see `preprocess_v2_DBLP.ipynb` section 1 for details (an extra `nltk` package is used in this section only).

Based on the cleaned dataset, we search schema instances (see `preprocess_v2_DBLP.ipynb` section 2) and save the data in `/data/DBLP_processed`.

We have uploaded all processed data in `/data/<dataset_name>_processed`. All raw datasets can be found [here](https://github.com/liuhao33/MSGNN/blob/main/data/readme.md).

To train the model and test it on link prediction task, please run

```shell
python run_DBLP.py --repeat 1 --patience 5
```

`--repeat`: the number of repeats.
`--patience`: the number of patience of early stopping.

### Ablation

To run ablation study, please use `--ablation` parameter.

`--ablation`: parameter of ablation study, choose one from {`None`, `graph`, `node`, `instance`, `global`}.

For example, to test the model without instance module, please run

```shell
python run_DBLP.py --repeat 1 --patience 5 --ablation instance
```

### Masking strategy

To test different masking strategies, please use `--masking-mode` parameter.

`--masking-mode`: masking strategy to be used, choose one from {`default`, `random`, `zero`}.

For example, to test the model with random masking strategy, please run

```shell
python run_DBLP.py --repeat 1 --patience 5 --masking-mode random
```

## Schema decomposition

The implementation of the proposed decomposition algorithm can be found in

```shell
schema_decomposition.ipynb
```

Please open and run all cells.

This script takes graph schema as input and outputs a decomposition plan.

Sevaral example schemas are provided in the script.

## Time consumption of instance search

To exam the time consumption of instance search, please open and run all cells

```shell
time_DBLP.ipynb
```

Direct search and parallel enabled search are both provided in the script.

### Sensitivity analysis

To run parameters sensitivity study, please use `--num-heads`, `--learning-rate`, `--nlayers` and `--dropout` parameters.

`--num-heads`: int, number of attention heads.

`--learning-rate`: float, learning rate.

`--nlayers`: int, number of MSGNN layers.

`--dropout`: float, dropout rate.

For example, to test the model with 6 heads, please run

```shell
python run_DBLP.py --repeat 1 --patience 5 --num-heads 6
```

The results of sensitivity analysis can be found in `sensitivity.xlsx`

### Generate representations for downstream tasks

MSGNN is a embedding model. To get node representations, please run

```shell
python representation_DBLP.py --save
```

`--save`: add this parameter to save representations.

The representations will be saved to `\data\DBLP_processed\representations` by default and can be utilized by other downstream tasks.
We use [HINormer](https://github.com/Ffffffffire/HINormer) as the classifer on node classifacation task.
