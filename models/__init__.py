from enum import Enum

from models.dnn import DNNDetector
from models.transformer import TransformerModel
from models.wavenet import Wave, ModelConfig


class NetworkType(Enum):
    DNN = 'DNN'
    WAVE = 'WAVE'
    Transformer = 'Transformer'


NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN: DNNDetector(),
                             NetworkType.WAVE: Wave(ModelConfig),
                             NetworkType.Transformer: TransformerModel()}
