# TriCoLo: Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval

This repository is the official implementation of [TriCoLo: Trimodal Contrastive Loss for Fine-grained Text to Shape Retrieval [project page]](https://3dlg-hcvc.github.io/tricolo/).

We introduce TriCoLo, a **tri**modal **co**ntrastive **lo**ss for text to 3D shape retrieval. Our network aligns text and shape embeddings in the hidden space such that the text representation can be used to retrieve matching objects.  Our model successfully grounds language describing shape, color and texture as well as handles negation.
We achieved SoTA on text to shape retrieval task and we hope our results will encourage more work on this direction.

## Usage

### Prerequisites

- Ubuntu 20.04.2
- GeForce RTX 2080 Ti
- CUDA 11.2
- Python 3.8
- Pytorch 1.10.1

### Setup

To install basic packages for this repo, run the following command.

```
git clone git@github.com:3dlg-hcvc/tricolo.git
cd tricolo
python -m pip install -r requirements.txt
```

To install CLIP, please refer to [CLIP[Github]](https://github.com/openai/CLIP)

To install Pytorch3d, please follow instructions in [pytorch3d/INSTALL.md](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)


## Dataset

We run our experiments on the "chairs and tables" dataset from [Text2shape](http://text2shape.stanford.edu/), which is subset of [ShapeNet](https://shapenet.org/) model that has been voxelized and annotated with text descriptions. 

To download the dataset, please
- Sign the terms-of-use for ShapeNet by going to the [ShapeNet webpage](https://shapenet.org/) and registering for access. Please make sure you sigh the terms-of-use for ShapeNet.
- Download voxelizations and text descriptions from [Text2shape](http://text2shape.stanford.edu/). The Text2Shape project provides multiple descriptions for each shape as well as surface and solid voxelization.  We use solid voxelization at 64x64x64 resolution for our experiments.
- Download rendered images. They are rendered by a [Blender-based script](https://github.com/panmari/stanford-shapenet-renderer).

Dataset should be put in a folder called `data` under the same directory where this repo is located.

## Training 

To train different variations of the model, run the following command with the appropriate expr_id

```
python run_retrieval.py --config_file tricolo/configs/retrieval_shapenet.yaml --expr_id v64i128b128
```
Argument `expr_id` is to specify the experiment. 
It guides the model to run Bimodal(I), Bimodal(V) or Trimodal(V+I) version. 

The following `expr_id` can be chosen:
- v64i128b128: Trimodal(V+I) experiment with voxel size 64, image size  128, batch size 128.
- v64b128: Bimodal(V) experiment with voxel size 64, batch size 128
- i128b128: Bimodal(I) experiment with image size 128, batch size 128
- tri_*: similar meaning but use triplet loss


## Evaluation

To evaluate the model on different data splits, run the following command with `--validate` (on validation set) or `--test` (on test set).

```
python run_retrieval_val.py --validate
```


```
python run_retrieval_val.py --test
```

### Calculate metrics

To calculate different metrics, run the following command with the path to the training results.

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

With pretrained checkpoints, our network can retrieve shapes according to your customized sentence. 

Change the customized sentence in `predict.py` and run the following command to retrieve a matching object:
```
python predict.py --customized_sentence circular_table_with_glass
```
The predicted object's 2D image and 3D nrrd file would be saved in `path/to/pretrained_checkpoints/predict`. 

### Visualization

To generate html files, run the following command with the path to the training results.

```
python tricolo/utils/gen_retrieval_result_html.py --root_dir logs/retrieval/path/to/nearest_neighbor_renderings/exp-date
```
The html files would be stored under `root_dir`.

# Acknowledgements

We thank [Eduardo Pontes Reis](https://github.com/edreisMD/ConVIRT-pytorch) for the [ConVIRT](https://arxiv.org/pdf/2010.00747.pdf) codebase, on which we build our repository.
