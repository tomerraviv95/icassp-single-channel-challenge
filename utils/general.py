import pickle


def load_pkl(filename):
    with open(f'{filename}.pickle', 'rb') as handle:
        data = pickle.load(handle)
    return data