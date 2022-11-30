from typing import Union, List
from dataclasses import dataclass

from src.interfaces import ModelInterface


@dataclass()
class Models:
    intended_models: List[ModelInterface]
    unintended_models: List[ModelInterface]
