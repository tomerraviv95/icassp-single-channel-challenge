from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder


class TransformerModel(nn.Module):
    def __init__(self):
        super(TransformerModel, self).__init__()
        self.d_model = 8
        encoder_layer = TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True)
        self.transformer = TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, y):
        reshaped_y = y.reshape(-1, self.d_model,8)
        reshaped_out = self.transformer(reshaped_y)
        out = reshaped_out.reshape(y.shape[0], -1)
        return out
