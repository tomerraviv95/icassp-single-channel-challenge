from enum import Enum

from models.bit_based.lstm import SignalAwareLSTM
from models.bit_based.signalawarednn import SignalAwareDNN
from models.bit_based_class import BitBasedWrapper
from models.signal_based.dnn import DNNDetector
from models.signal_based.transformer import TransformerModel
from models.signal_based.wavenet import Wave
from models.signal_based_class import SignalBasedWrapper


class NetworkType(Enum):
    DNN = 'DNN'
    Transformer = 'Transformer'
    WAVE = 'WAVE'
    SignalAwareDNN = 'SignalAwareDNN'
    SignalAwareLSTM = 'SignalAwareLSTM'


NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN: DNNDetector,
                             NetworkType.Transformer: TransformerModel,
                             NetworkType.WAVE: Wave,
                             NetworkType.SignalAwareDNN: SignalAwareDNN,
                             NetworkType.SignalAwareLSTM: SignalAwareLSTM}


class WrapperType(Enum):
    ModelFree = 'ModelFree'
    ModelBased = 'ModelBased'


TYPES_TO_WRAPPER = {WrapperType.ModelFree: SignalBasedWrapper,
                    WrapperType.ModelBased: BitBasedWrapper}
