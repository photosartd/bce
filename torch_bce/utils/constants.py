import enum


class ModelType(enum.Enum):
    NOT_DEFINED = 0
    REGRESSION = 1
    CLASSIFICATION = 2
    UNSUPERVISED = 3
