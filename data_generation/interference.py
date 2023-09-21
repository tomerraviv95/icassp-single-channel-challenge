import os

import h5py
import numpy as np

from globals import DATA_FOLDER


def load_interference(interference_sig_type):
    with h5py.File(os.path.join(DATA_FOLDER, 'interferenceset_frame', interference_sig_type + '_raw_data.h5'),
                   'r') as data_h5file:
        sig_data = np.array(data_h5file.get('dataset'))
        sig_type_info = data_h5file.get('sig_type')[()]
        if isinstance(sig_type_info, bytes):
            sig_type_info = sig_type_info.decode("utf-8")
    return sig_data, sig_type_info
