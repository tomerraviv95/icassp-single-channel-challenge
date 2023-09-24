from enum import Enum

from data_generation.dataset import SINR_values
from models.dnn import DNNDetector
from models.transformer import TransformerModel
from models.wavenet import Wave, ModelConfig


class NetworkType(Enum):
    DNN = 'DNN'
    WAVE = 'WAVE'
    Transformer = 'Transformer'


NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN: DNNDetector(),
                             NetworkType.Transformer: TransformerModel(),
                             NetworkType.WAVE: Wave(ModelConfig)}

def initialize_networks(model_type):
    nets = [NETWORKS_TYPES_TO_METHODS[model_type] for _ in range(len(SINR_values))]
    return nets
