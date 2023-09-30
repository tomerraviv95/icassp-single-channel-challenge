from enum import Enum

from models.model_based_class import ModelBasedWrapper
from models.model_free.dnn import DNNDetector
from models.model_free.transformer import TransformerModel
from models.model_free.wavenet import Wave
from models.model_free_class import ModelFreeWrapper
from models.viterbinet import ViterbiNet


class NetworkType(Enum):
    DNN = 'DNN'
    Transformer = 'Transformer'
    WAVE = 'WAVE'
    ViterbiNet = 'ViterbiNet'


NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN: DNNDetector,
                             NetworkType.Transformer: TransformerModel,
                             NetworkType.WAVE: Wave,
                             NetworkType.ViterbiNet: ViterbiNet}


class WrapperType(Enum):
    ModelFree = 'ModelFree'
    ModelBased = 'ModelBased'


TYPES_TO_WRAPPER = {WrapperType.ModelFree: ModelFreeWrapper,
                    WrapperType.ModelBased: ModelBasedWrapper}
