# TriCoLo: Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval

This repository is the official implementation of [TriCoLo: Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval [project page]](https://3dlg-hcvc.github.io/tricolo/).

## Usage

### Prerequisites

- Linux (not tested for MacOS or Windows)
- Python3.7
- CUDA version 11.2
- Pytorch version 1.10.1

### Setup

```
git clone git@github.com:3dlg-hcvc/tricolo.git
cd tricolo
python -m pip install -r requirements.txt
```

## Dataset

- [Text2shape Dataset](http://text2shape.stanford.edu/). It provides fine-grained texts and Shapenet voxels among which we use solid voxelization: 64 Resolution.
- Corresponding Images. They are rendered by a [Blender-based script](https://github.com/panmari/stanford-shapenet-renderer).

Dataset should be put in a folder called `data` under the same directory where this repo is located.

## Training 
```
python run_retrieval.py --config_file tricolo/configs/retrieval_shapenet.yaml --param_id X
```
X is the number showing which setting are used here. It guides the model to run Bimodal(I), Bimodal(V) or Trimodal(V+I) version. Details are in `run_retrieval.py`.

## Evaluation

### Evaluate model on VALIDATION set

```
python run_retrieval_val.py --validate
```

### Evaluate model on TEST set

```
python run_retrieval_val.py --test
```

### Calculate metrics
- L2 Chamfer distance
- Normal consistency
- Absolute normal consistency
- Precision at various thresholds
- Recall at various thresholds
- F1 score at various thresholds

```
python tricolo/metrics/compare_meshes.py --prediction_dir logs/retrieval/exp-date/test/nearest_neighbor_renderings/another-exp-date
```

### Inference

Via pretrained checkpoint, this neural network can retrieve shapes according to your customized sentence. Detains are in `predict.py`


### Visualization

Generate html files:

```
python tricolo/utils/gen_retrieval_result_html.py --root_dir logs/retrieval/path/to/nearest_neighbor_renderings/exp-date
```
The html files would be stored under `root_dir`.

# Acknowledgements

We thank [Eduardo Pontes Reis](https://github.com/edreisMD/ConVIRT-pytorch) for the [ConVIRT](https://arxiv.org/pdf/2010.00747.pdf) codebase, on which we build our repository.