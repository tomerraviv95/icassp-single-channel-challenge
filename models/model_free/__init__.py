from enum import Enum

from models.model_free.dnn import DNNDetector
from models.model_free.transformer import TransformerModel
from models.model_free.wavenet import Wave


class NetworkType(Enum):
    DNN = 'DNN'
    Transformer = 'Transformer'
    WAVE = 'WAVE'


NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN: DNNDetector,
                             NetworkType.Transformer: TransformerModel,
                             NetworkType.WAVE: Wave}
