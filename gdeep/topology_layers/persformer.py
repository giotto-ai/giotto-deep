# from https://github.com/juho-lee/set_transformer/blob/master/max_regression_demo.ipynb  # noqa: E501
#### Author: Raphael Reinauer
import torch  # type: ignore
import torch.nn as nn  # type: ignore
from torch import Tensor  # type: ignore
import torch.nn.functional as F
from torch.nn import (Module, Linear)
from gdeep.topology_layers.modules import ISAB, PMA, SAB  # type: ignore


    
    
class Persformer(Module):
    """
    Persformer architecture as described in
    "Persformer: A Transformer Architecture for Topological Machine Learning"
    https://arxiv.org/abs/2112.15210
    

    Args:
        dim_input (int, optional):
            Dimension of input data for each element in the set. Defaults to 4.
        dim_output (int, optional):
            Output dimension of the model. This corresponds to the number of classes. Defaults to 5.
        dim_hidden (int, optional):
            Hidden dimension of the encoder. Defaults to 128.
        num_heads (str, optional):
            Number of attention heads in every attention layer. Defaults to "4".
        layer_norm (str, optional):
            Use layer normalization in the encoder. Defaults to "False".
        pre_layer_norm (str, optional):
            Use pre-layer normalization. Defaults to "False".
        simplified_layer_norm (str, optional):
           Use simplified layer normalization.
           See
           https://proceedings.neurips.cc/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
           Defaults to "True".
        dropout_enc (float, optional):
            Dropout probability in the encoder. Defaults to 0.0.
        dropout_dec (float, optional):
            Dropout probability in the decoder. Defaults to 0.0.
        num_layer_enc (int, optional):
            Number of attention blocks in the encoder. Defaults to 2.
        num_layer_dec (int, optional):
            Number of feed-forwards layers in the decoder. Defaults to 3.
        activation (str, optional):
            Activation function to use in the whole model. Defaults to "gelu".
        bias_attention (str, optional):
            Use bias in the query, key and value computation. Defaults to "True".
        attention_type (str, optional):
            Either use full self-attention with quadratic complexity (´self_attention´, ´pytorch_self_attention´)
            full self-attention with skip-connections (´pytorch_self_attention_skip´),
            or induced attention with linear complexity (´induced_attention´). Defaults to "´pytorch_self_attention_skip´".
        layer_norm_pooling (str, optional):
            Use layer norm in the multi-head attention pooling layer. Defaults to "False".

    Raises:
        ValueError:
            [description]
    """
    def __init__(
        self,
        dim_input=4,  # 
        num_outputs=1,
        dim_output=5,
        num_inds=32,  # number of induced points, see  Set Transformer paper
        dim_hidden=128,
        num_heads="4",
        layer_norm="False",  # use layer norm
        pre_layer_norm="False", # use pre-layer norm
        simplified_layer_norm="True",
        dropout_enc=0.0,
        dropout_dec=0.0,
        num_layer_enc=2,
        num_layer_dec=3,
        activation="gelu",
        bias_attention="True",
        attention_type="´pytorch_self_attention_skip´",
        layer_norm_pooling="False",
    ):
        super().__init__()
        self._attention_type = attention_type
        bias_attention = eval(bias_attention)
        if activation == 'gelu':
            activation_layer = nn.GELU()
            activation_function = F.gelu
        elif activation == 'relu':
            activation_layer = nn.ReLU()
            activation_function = F.relu
        else:
            raise ValueError("Unknown activation '%s'" % activation)
        
        if attention_type=="induced_attention":
            self.enc = nn.Sequential(
                ISAB(dim_input, dim_hidden, eval(num_heads), num_inds, ln=eval(layer_norm),
                        simplified_layer_norm = eval(simplified_layer_norm),
                        bias_attention=bias_attention, activation=activation),
                *[ISAB(dim_hidden, dim_hidden, eval(num_heads), num_inds, ln=eval(layer_norm),
                        simplified_layer_norm = eval(simplified_layer_norm),
                        bias_attention=bias_attention, activation=activation)
                    for _ in range(num_layer_enc-1)],
            )
        elif attention_type=="self_attention":
            self.enc = nn.Sequential(
                SAB(dim_input, dim_hidden, eval(num_heads), ln=eval(layer_norm),
                    simplified_layer_norm = eval(simplified_layer_norm),
                    bias_attention=bias_attention, activation=activation),
                *[SAB(dim_hidden, dim_hidden, eval(num_heads), ln=eval(layer_norm),
                        simplified_layer_norm = eval(simplified_layer_norm),
                        bias_attention=bias_attention, activation=activation)
                    for _ in range(num_layer_enc-1)],
            )
        elif attention_type=="pytorch_self_attention":
            emb = Linear(dim_input, dim_hidden)
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim_hidden,
                                                    nhead=eval(num_heads),
                                                    dropout=dropout_enc,
                                                    activation=activation_function,
                                                    norm_first=eval(pre_layer_norm),
                                                    batch_first=True)
            self.enc = nn.Sequential(
                    emb,
                    nn.TransformerEncoder(encoder_layer,
                                                num_layers=num_layer_enc)
            )
        elif attention_type=="pytorch_self_attention_skip":
            self.emb = Linear(dim_input, dim_hidden)
            self.encoder_layers = nn.ModuleList(
                [nn.TransformerEncoderLayer(d_model=dim_hidden,
                                                    nhead=eval(num_heads),
                                                    dropout=dropout_enc,
                                                    activation=activation_function,
                                                    norm_first=eval(pre_layer_norm),
                                                    batch_first=True) for _ in range(num_layer_enc)]
            )
        else:
            raise ValueError("Unknown attention type: {}".format(attention_type))
        enc_layer_dim = [2**i if i <= num_layer_dec/2 else num_layer_dec - i for i in range(num_layer_dec)]
        self.dec = nn.Sequential(
            nn.Dropout(dropout_dec),
            PMA(dim_hidden, eval(num_heads), num_outputs, ln=eval(layer_norm_pooling),
                simplified_layer_norm = eval(simplified_layer_norm),
                bias_attention=bias_attention, activation=activation),
            nn.Dropout(dropout_dec),
            *[nn.Sequential(nn.Linear(enc_layer_dim[i] * dim_hidden, enc_layer_dim[i+1] * dim_hidden),
                            activation_layer,
                            nn.Dropout(dropout_dec)) for i in range(num_layer_dec-1)],
            nn.Linear(enc_layer_dim[-1] * dim_hidden, dim_output),
        )

    def forward(self, input):
        if self._attention_type == "pytorch_self_attention_skip":
            x = self.emb(input)
            for l in self.encoder_layers:
                x = x + l(x)
            return self.dec(x).squeeze(dim=1)
        else:
            return self.dec(self.enc(input)).squeeze(dim=1)

    @property
    def num_params(self) -> int:
        """Returns number of trainable parameters.
        Returns:
            int: Number of trainable parameters.
        """
        total_params = 0
        for parameter in self.parameters():
            total_params += parameter.nelement()
        return total_params
    
    

class GraphClassifier(nn.Module):
    """Classifier for Graphs using persistence features and additional
    features. The vectorization is based on a set transformer.
    """
    def __init__(self,
                 num_features,
                 dim_input=6,
                 num_outputs=1,
                 dim_output=50,
                 num_classes=2,
                 ln=True,
                 num_heads=4,
                 use_induced_attention=False,
                 dim_hidden=128,
                 dropout=0.0,
                ):
        super().__init__()
        self.st = Persformer(
            dim_input=dim_input,
            num_outputs=num_outputs,
            dim_output=dim_output,
            ln=ln,
            num_heads=num_heads,
            dim_hidden=dim_hidden,
            use_induced_attention=use_induced_attention,
            dropout=dropout,
            )
        self.num_classes = num_classes
        self.ln = nn.LayerNorm(dim_output + num_features)
        self.ff_1 = nn.Linear(dim_output + num_features, 50)
        self.ff_2 = nn.Linear(50, 20)
        self.ff_3 = nn.Linear(20, num_classes)
    def forward(self, x_pd: Tensor, x_feature: Tensor) -> Tensor:
        """Forward pass of the graph classifier.
        The persistence features are encoded with a set transformer
        and concatenated with the feature vector. These concatenated
        features are used for classification using a fully connected
        feed -forward layer.
        
        Args:
            x_pd (Tensor):
                persistence diagrams of the graph
            x_feature (Tensor):
                additional graph features
        """
        pd_vector = self.st(x_pd)
        features_stacked = torch.hstack((pd_vector, x_feature))
        x = self.ln(features_stacked)
        x = nn.ReLU()(self.ff_1(x))
        x = nn.ReLU()(self.ff_2(x))
        x = self.ff_3(x)
        return x