# HAPC-Net

Skeleton-based isolated sign language recognition. The network takes a sequence of 27 body and hand keypoints and predicts the sign class.

Each block splits its channels into a spatial branch, a graph convolution whose adjacency combines a learned static graph, two sample-dependent graphs and a hypergraph with virtual joints, and a temporal branch of multi-scale temporal convolutions. During training, an auxiliary prototype branch adds two contrastive losses; it is skipped at inference time.

## Installation

The code was developed with Python 3.10 and PyTorch 2.4.1 (CUDA 12.1).

```bash
conda create -n hapc python=3.10
conda activate hapc
```

Install PyTorch for your CUDA version by following [pytorch.org](https://pytorch.org/get-started/locally/), then the remaining packages:

```bash
pip install -r requirements.txt
```

## Data preparation

The code uses the preprocessed 27-keypoint skeleton data released with [DSTA-SLR](https://github.com/hulianyuyy/DSTA-SLR), available [here](https://mega.nz/folder/EvkEzIAC#gq_nWLbbWoj9WVnJGxnGaA). Please follow the rules and agreements of the original datasets when using it.

Place, or link, each dataset under `./data`:

```
data/
└── WLASL2000/
    ├── train_data_joint.npy
    ├── train_label.pkl
    ├── val_data_joint.npy
    ├── val_label.pkl
    ├── test_data_joint.npy
    └── test_label.pkl
```

```bash
ln -s path_to_your_WLASL2000/WLASL2000/ ./data/WLASL2000
```

The provided configs cover WLASL100, WLASL300, WLASL2000 and MSASL1000.

## Training

```bash
python main.py --config config/train/train.yaml --device 0
```

The config trains the joint stream on WLASL2000. Everything a run produces is written to `./work_dir/<Experiment_name>/`: a copy of the config and code, `log.txt`, and the best checkpoint in `save_models/best_model.pt`. The model is evaluated on the validation split every `save_interval` epochs.

To train on another dataset, change `dataset` and `model_args.num_classes` in [config/train/train.yaml](config/train/train.yaml):

| dataset | num_classes |
|---|---|
| WLASL100 | 100 |
| WLASL300 | 300 |
| WLASL2000 | 2000 |
| MSASL1000 | 1000 |

## Testing

```bash
python main.py --config config/test/test.yaml --device 0
```

In [config/test/test.yaml](config/test/test.yaml), set `weights` to the checkpoint, and `dataset`, `model_args.num_classes`, `data_path` and `label_path` to the dataset it was trained on. `model_args`, `bone_stream` and `motion_stream` must match the training run.

Besides printing the accuracy, the test run writes the class scores of every sample to `./work_dir/<Experiment_name>/eval_results/<score_name>`, which is the input to the ensemble.

## Multi-stream ensemble

The four streams are selected with two flags, which must be set identically in `train_feeder_args` and `test_feeder_args`:

| stream | bone_stream | motion_stream |
|---|---|---|
| joint | False | False |
| bone | True | False |
| joint motion | False | True |
| bone motion | True | True |

1. Train one model per stream, giving each run its own `Experiment_name`.
2. Run the test config once per stream to write its score file.
3. Set the paths of the label file and of the four score files, and the fusing weights `alpha`, at the top of [ensemble/ensemble.py](ensemble/ensemble.py), then run it from inside the folder:

```bash
cd ensemble
python ensemble.py
```

To choose the fusing weights, write the score files for the validation split instead (point `data_path` and `label_path` of the test config to the validation files), set their paths in [ensemble/ensemble_search.py](ensemble/ensemble_search.py) and run it the same way. It grid-searches the four weights and prints the best combinations.

## Repository structure

```
main.py              training and evaluation
config/              train and test configs
model/model.py       the network
model/               its building blocks and the auxiliary losses
feeders/             dataset and augmentation
ensemble/            multi-stream fusion
```

## Acknowledgements

This code builds on [DSTA-SLR](https://github.com/hulianyuyy/DSTA-SLR), [SkateFormer](https://github.com/KAIST-VICLab/SkateFormer), [SAM-SLR-v2](https://github.com/jackyjsy/SAM-SLR-v2) and [SLGTformer](https://github.com/neilsong/SLGTformer). Many thanks to the authors for open-sourcing their work.

## License

Released under the Attribution-NonCommercial 4.0 International license; see [LICENSE](LICENSE).
