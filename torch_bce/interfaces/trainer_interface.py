import logging
import os.path
import pickle as pkl
from typing import List, Dict, Union, Tuple
from abc import ABC, abstractmethod

import wandb
import torch.optim


from torch_bce.interfaces import Saveable, ModelInterface
from torch_bce.utils.datasets import TensorDataset
from torch_bce.containers import ListModelContainer
from torch_bce.losses import AlignmentLoss


class TrainerInterface(Saveable, ABC):
    """Abstract class created for easy training of models
        Implements template method `train`
        Implements template method `replace_model`
        Implements template method `inference`
        Parameters should be prepared in setup stage
        Parameters:
            :param models: models (of subtype ModelInterface)
            :param device: where to train nets
            :param logger: logging.Logger
        """
    models: ListModelContainer
    logger: logging.Logger
    device: torch._C.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __init__(self, *args, **kwargs):
        self.setup_logger(**kwargs)
        self.logger = logging.getLogger("default_logger")
        self.kwargs = kwargs

    def setup(self, **kwargs) -> None:
        """Implement all basic operations that are needed
        Arguments:
            :param kwargs: any keyword arguments you want to supply to 'configure_*' methods
            :return: None
        """
        self.models = self.configure_models(**kwargs)

    @staticmethod
    def setup_logger(name="default_logger", level=logging.WARN, **kwargs) -> None:
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logging.basicConfig()

    @abstractmethod
    def log_metrics(self, **kwargs):
        """Make all needed logging"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    def configure_models(**kwargs) -> ListModelContainer:
        """Returns torch_bce.containers.ListModelContainer"""
        raise NotImplementedError()

    @abstractmethod
    def train(self, *args, **kwargs):
        """Trains current model"""
        raise NotImplementedError()

    @abstractmethod
    def replace_model(self, model: ModelInterface) -> None:
        """Replaces current model with new for training"""
        raise NotImplementedError()

    @abstractmethod
    def inference(self, dataloader):
        """Makes inference on dataloader and returns torch.Tensor"""
        raise NotImplementedError()

    def save(self, savepath: str, **kwargs) -> None:
        models_path = f"{savepath}/models"
        os.mkdir(models_path)
        self.models.save(models_path, **kwargs)
        with open(os.path.join(savepath, "kwargs.pkl"), "wb") as kw_file:
            pkl.dump(self.kwargs, kw_file)

    @classmethod
    def load(cls, path, **kwargs):
        """Loads trainer with all models
        Kwargs:
            :class_type: ModelInterface or relatives
        """
        models_path = f"{path}/models"
        models = ListModelContainer.load(models_path, **kwargs)
        with open(os.path.join(path, "kwargs.pkl"), "rb") as kw_file:
            old_kwargs = pkl.load(kw_file)
        if isinstance(kwargs, dict):
            old_kwargs.update(**kwargs)
        return cls(model=models, **old_kwargs)


class AlignmentTrainerInterface(TrainerInterface, ABC):
    alignment: AlignmentLoss
    alignment_optimizer: torch.optim.Optimizer
    current_model: ModelInterface

    def __init__(self, prev_embeddings: TensorDataset = None, *args, **kwargs):
        super(AlignmentTrainerInterface, self).__init__(*args, **kwargs)
        self.setup(**kwargs)
        self.current_model = self.models[-1]
        self.alignment = self.configure_alignment(**kwargs)
        self.alignment_optimizer = self.configure_alignment_optimizer(**kwargs)
        """prev_embeddings Tensor dummy initialization"""
        self.prev_embeddings: TensorDataset = \
            TensorDataset(x=torch.rand(())) if prev_embeddings is None else prev_embeddings

    def setup(self, num_epochs=100, setup_wandb=True, **kwargs) -> None:
        super(AlignmentTrainerInterface, self).setup(**kwargs)
        # TODO: key to secrets
        if setup_wandb:
            wandb.login(key="ff032b7d576dfc8849b5c26d72c084cfc88f7b76")
            wandb.init(project="bce", reinit=True)
            config = wandb.config
            config.epochs = num_epochs

    def log_metrics(self, epoch: int, metrics: Dict, **kwargs):
        wandb.log(metrics)
        for metric_name, metric_val in metrics.items():
            self.logger.info(f"Epoch: {epoch}| Metric {metric_name}: {metric_val}")

    def configure_models(self, model: Union[ModelInterface, ListModelContainer] = None, **kwargs) -> ListModelContainer:
        if isinstance(model, ListModelContainer):
            model.models = [m.to(self.device) for m in model.models]
            return model
        elif isinstance(model, ModelInterface):
            model = model.to(self.device)
            return ListModelContainer([model])
        else:
            raise TypeError("Model must be of of type Union[ModelInterface, ListModelContainer]")

    def configure_alignment(self, lambda_=4, alignment="single_step", backward_transformation="linear",
                            alignment_loss: AlignmentLoss = None, **kwargs):
        """Configures alignment for model training"""
        if alignment_loss is not None and isinstance(alignment_loss, AlignmentLoss):
            return alignment_loss
        M = self.models[-1].out_channels
        N = M
        alignment_loss = AlignmentLoss(
            lambda_=lambda_,
            backward_transformation=backward_transformation,
            alignment=alignment,
            M=M,
            N=N
        )
        alignment_loss.to(self.device)
        return alignment_loss

    def configure_alignment_optimizer(self, lr: float = 3e-1, weight_decay: float = 4e-2,
                                      **kwargs) -> torch.optim.Optimizer:
        """Needs to be called every time in `replace_model`"""
        optimizer = torch.optim.Adam(self.alignment.parameters(), lr=lr, weight_decay=weight_decay)
        return optimizer

    def train(self,
              train_data: object,
              val_data: Union[object, None],
              *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Trains model
        Arguments:
            :param train_data: some data that can be passed to model
            :param val_data: some data that can be passed to model
        Returns:
            model_loss: needs to be optimized
            statistics: dict with stats
        """
        """Training"""
        self.alignment_optimizer.zero_grad()
        model_loss, alignment_loss, statistics = self.train_model(train_data, *args, **kwargs)
        alignment_loss.backward(retain_graph=True)
        self.alignment_optimizer.step()

        """Validation"""
        predictions, val_model_loss, val_alignment_loss = self.validate_model(val_data, *args, **kwargs)
        statistics = {
            "model_loss": statistics["loss"],
            "val_model_loss": val_model_loss.item() if isinstance(val_model_loss, torch.Tensor) else val_model_loss,
            "alignment_loss": statistics["external_loss"],
            "val_alignment_loss": val_alignment_loss.item() if isinstance(val_alignment_loss,
                                                                          torch.Tensor) else val_alignment_loss
        }
        return model_loss, statistics

    @abstractmethod
    def train_model(self, train_data: object, *args, **kwargs) -> \
            Tuple[torch.Tensor, torch.Tensor, Dict]:
        """Returns model loss and alignment loss"""
        raise NotImplementedError()

    @abstractmethod
    def validate_model(self, val_data: object, *args, **kwargs) -> \
            Tuple[torch.Tensor, Union[torch.Tensor, None], Union[torch.Tensor, None]]:
        """Returns predictions, model loss and alignment loss if val_data is not none, else None"""
        raise NotImplementedError()

    @abstractmethod
    def initialize_new_model(self, prev_model: ModelInterface, new_model: ModelInterface, *args, **kwargs) -> None:
        """Initializes new model layers with old model where (if) it's possible"""
        raise NotImplementedError()

    def inference(self, data: object, ver: int = -1, *args, **kwargs) -> Tuple[torch.Tensor, Dict]:
        """Makes inference on dataloader and returns:
        :return
            :predictions: torch.Tensor
            :statistics: dict
        """
        predictions, val_model_loss, val_alignment_loss = self.validate_model(data, *args, **kwargs)
        predictions = self.alignment.inference(predictions, ver)
        statistics = dict(
            val_loss_model=val_model_loss.item() if isinstance(val_model_loss, torch.Tensor) else val_model_loss,
            val_loss_alignment=val_alignment_loss.item() if isinstance(val_alignment_loss,
                                                                       torch.Tensor) else val_alignment_loss
        )
        return predictions, statistics

    def alignment_loss(self, m_k: torch.Tensor, n_id: List[int]):
        """Calculates alignment loss for current intended model
        Arguments:
            :param m_k: tensor with new embeddings
            :param n_id: ID's of nodes for m_k (respective to their initial place)
        """
        if len(self.models) == 1:
            return torch.zeros((), requires_grad=True)
        else:
            old_embeddings = self.prev_embeddings[n_id[:m_k.shape[0]]]
            assert m_k.shape[0] == old_embeddings.shape[0], \
                f"m_k shape: {m_k.shape}, old shape: {old_embeddings.shape}"
            return self.alignment(m_k, old_embeddings)

    def replace_model(self, model: ModelInterface, train_data: object = None, **kwargs) -> None:
        """Replaces model
        Arguments:
            :param model: new model for training
            :param train_data: train dataset for new embedding creation
        Possible kwargs arguments:
            :param lr: alignment_optimizer lr
            :param weight_decay: alignment_optimizer weight_decay
            :param step_size: StepLR parameter
            :param gamma: StepLR parameter
        """
        """TODO: Check embeddings are in the same order"""
        if train_data is None:
            self.logger.warning("train_data in `replace_model` must not be None")
        else:
            predictions, val_model_loss, val_alignment_loss = self.validate_model(train_data, **kwargs)
            self.prev_embeddings = TensorDataset(x=predictions.to(self.device))

        prev_model = self.current_model
        self.models.add(model)
        self.current_model = model
        self.initialize_new_model(prev_model, self.current_model)
        self.alignment.finish(M=self.current_model.out_channels,
                              N=prev_model.out_channels, **kwargs)
        self.alignment_optimizer = self.configure_alignment_optimizer(**kwargs)

    def save(self, savepath, **kwargs) -> None:
        super(AlignmentTrainerInterface, self).save(savepath, **kwargs)
        with open(f"{savepath}/alignment.pkl", "wb") as f:
            pkl.dump(self.alignment, f)
        with open(f"{savepath}/prev_embeddings.pkl", "wb") as f:
            pkl.dump(self.prev_embeddings, f)

    @classmethod
    def load(cls, path, **kwargs):
        """Loads trainer with all models
        Kwargs:
            :class_type: ModelInterface or relatives
        """
        models_path = f"{path}/models"
        models = ListModelContainer.load(models_path, **kwargs)
        with open(os.path.join(path, "kwargs.pkl"), "rb") as kw_file:
            old_kwargs = pkl.load(kw_file)
        with open(f"{path}/alignment.pkl", "rb") as f:
            alignment_loss = pkl.load(f)
        with open(f"{path}/prev_embeddings.pkl", "rb") as f:
            prev_embeddings = pkl.load(f)
        if isinstance(kwargs, dict):
            old_kwargs.update(**kwargs)
            old_kwargs["alignment_loss"] = alignment_loss
            old_kwargs["prev_embeddings"] = prev_embeddings
            old_kwargs["model"] = models
        return cls(**old_kwargs)
