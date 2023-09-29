from enum import Enum

from models.model_free_class import ModelFreeWrapper


class WrapperType(Enum):
    ModelFree = 'ModelFree'
    ModelBased = 'ModelBased'


TYPES_TO_WRAPPER = {WrapperType.ModelFree: ModelFreeWrapper}
