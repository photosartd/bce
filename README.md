# Studying Backward Compatible Embeddings
A repository created for studying different aspects of [Backward Compatible Embeddings](https://arxiv.org/abs/2206.03040)
DL architecture for machine learning tasks.
## Problem formulation
**Problem formulation** can be found in the original paper repository [bc-emb](https://github.com/snap-stanford/bc-emb).
## Modules
`torch_bce` contains several main modules:
1. `containers`
   1. `ListModelContainer` - container for used models
   2. `WeightsContainer` - module for **Backward Transformation** storage, `@deprecated(0.1.0)`
2. `interfaces` - main interfaces to inherit from
   1. `ModelInterface` - interface for PyTorch backward compatible models to subclass
   2. `Saveable` - save/load interface
   3. `TrainerInterface` - interface for trainer
3. `losses` - different useful losses.
   1. `GraphSAGELoss` - loss for unsupervised embeddings learning on graphs as described in [paper](https://arxiv.org/pdf/1706.02216.pdf).
   2. `AlignmentLoss` - loss for alignment of different embeddings that has its own state and should be optimized.
   3. `BackwardTransformation` - backward transformation for embeddings.
4. `models`
   1. `intended`
      1. `GraphSAGE`
   2. `unintended`
      1. `MLP`
5. `trainers`
   1. `GSAlignmentTrainer` - for training of intended models in unified manner.
6. `tests` - tests with help of [unittest](https://docs.python.org/3/library/unittest.html)
7. `utils` - some useful utilities, such as:
   1. `datasets`
   2. `metrics`
   3. `samplers`
## Classes Diagram
![](./images/UML%20Backward%20Compatible%20Embeddings%20Architecture.png)
## Reproducibility
### Docker
For now there is [Dockerfile](./Dockerfile) that enables to start all tests. Just execute `./build_test.sh` from 
inside the directory.
### Setup
To set up the framework, execute `pip install -e .` from inside the directory. There is also [requirements](./requirements.txt)
file, however some dependencies (e.g., **PyTorch Geometric** and **PyTorch Geometric Temporal** must be installed manually,
as it is done in Dockerfile.
### Example
<a target="_blank" href="https://colab.research.google.com/github/photosartd/bce/blob/dev/bce_example.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
