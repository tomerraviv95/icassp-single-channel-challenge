from enum import Enum

from models.bit_based.signalawarelstm import SignalAwareLSTM
from models.bit_based.signalawarednn import SignalAwareDNN
from models.bit_based_class import BitBasedWrapper
from models.mixed.mixed import MixedLSTM
from models.mixed_class import MixedWrapper
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
    MixedLSTM = 'MixedLSTM'


NETWORKS_TYPES_TO_METHODS = {NetworkType.DNN: DNNDetector,
                             NetworkType.Transformer: TransformerModel,
                             NetworkType.WAVE: Wave,
                             NetworkType.SignalAwareDNN: SignalAwareDNN,
                             NetworkType.SignalAwareLSTM: SignalAwareLSTM,
                             NetworkType.MixedLSTM: MixedLSTM}


class WrapperType(Enum):
    SignalBased = 'SignalBased'
    BitBased = 'BitBased'
    Mixed = 'Mixed'


TYPES_TO_WRAPPER = {WrapperType.SignalBased: SignalBasedWrapper,
                    WrapperType.BitBased: BitBasedWrapper,
                    WrapperType.Mixed: MixedWrapper}
