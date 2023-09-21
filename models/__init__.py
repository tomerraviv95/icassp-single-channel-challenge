from enum import Enum

from models.dnn import DNNDetector


class NetworkType(Enum):
    DNN = 'DNN'

NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN:DNNDetector}

