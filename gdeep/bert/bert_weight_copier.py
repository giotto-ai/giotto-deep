"""
Copy weights of pretrained BERT to our simplified BERT

We use a pretrained BERT model from Huggingface
(https://huggingface.co/transformers/model_doc/bert.html)
and copy the weights to our simplified BERT model defined
in gdeep.simplified_bert_model.
For simplicity we will call the BERT model from huggingface
hbert and our simplified bert model sbert.

Hacked by Raphael Reinauer 2021 (https://github.com/raphaelreinauer)
"""


import torch
import torch.nn as nn

from transformers import BertConfig, BertLayer, BertModel  # type: ignore
from transformers.modeling_bert import BertSelfAttention  # type: ignore

from .simplified_bert_model import SimplifiedMultiHeadSelfAttention, \
    SimplifiedBertBlock, SimplifiedBertEncoder, SimplifiedBertEmbeddings, \
    SimplifiedBertClassifier


class BertSelfAttentionWeightCopier():
    def __init__(self,
                 bert_self_attention_layer: BertSelfAttention,
                 config: BertConfig
                 ) -> None:
        self.config = config

        # stack weights and biases of bert_self_attention_layer
        self.stacked_weights = torch.cat(
            [bert_self_attention_layer.state_dict()[key]
                for key in ['query.weight', 'key.weight', 'value.weight']],
            dim=0
        )
        self.stacked_biases = torch.cat(
            [bert_self_attention_layer.state_dict()[key]
                for key in ['query.bias', 'key.bias', 'value.bias']],
            dim=0
        )

    def build_layer(self) -> SimplifiedMultiHeadSelfAttention:
        # initialize multi head self attention layer
        mhsa_layer = SimplifiedMultiHeadSelfAttention(self.config)

        # set stacked weights and biases from bert_self_attention_layer as the
        # weights of the mhsa_layer
        with torch.no_grad():
            mhsa_layer.to_qvk.weight = nn.Parameter(self.stacked_weights)
            mhsa_layer.to_qvk.bias = nn.Parameter(self.stacked_biases)

        return mhsa_layer


class BertBlockWeightCopier():
    def __init__(self,
                 bert_layer: BertLayer,
                 config: BertConfig
                 ) -> None:
        self.config = config
        self.bert_layer = bert_layer

    def build_layer(self) -> SimplifiedBertBlock:
        simplified_bert_block = SimplifiedBertBlock(self.config)

        # copy mhsa block
        mhsa_builder = BertSelfAttentionWeightCopier(
            self.bert_layer.attention.self,
            self.config
            )
        bert_mhsa = mhsa_builder.build_layer()
        simplified_bert_block.mhsa = bert_mhsa

        # dictionary of layer of the bert blocks
        # that correspond to each other
        # the keys are to the sbert layers and values are the
        # corresponding hbert layers
        layer_correspondence = {
            simplified_bert_block.attention_linear:
                self.bert_layer.attention.output.dense,
            simplified_bert_block.norm_1:
                self.bert_layer.attention.output.LayerNorm,
            simplified_bert_block.linear[2]:
                self.bert_layer.output.dense,
            simplified_bert_block.norm_2:
                self.bert_layer.output.LayerNorm,
            simplified_bert_block.linear[0]:
                self.bert_layer.intermediate.dense
            }

        # copy weights and biases from hbert to sbert
        for simplified_layer, huggingface_layer\
                in layer_correspondence.items():
            simplified_layer.weight = nn.Parameter(
                                            huggingface_layer.weight
                                        )
            simplified_layer.bias = nn.Parameter(
                                            huggingface_layer.bias
                                        )

        return simplified_bert_block


class BertEncoderWeightCopier():
    def __init__(self,
                 bert_model: BertModel,
                 config: BertConfig
                 ) -> None:
        self.bert_model = bert_model
        self.config = config
        self.blocks = config.num_hidden_layers

    def build_layer(self) -> SimplifiedBertEncoder:
        simplified_bert_encoder = SimplifiedBertEncoder(self.config)
        for layer_idx in range(self.blocks):
            bbc = BertBlockWeightCopier(
                self.bert_model.bert.encoder.layer[layer_idx],
                self.config
                )
            bert_block = bbc.build_layer()
            simplified_bert_encoder.layers[layer_idx] = bert_block

        return simplified_bert_encoder


class BertEmbeddingsWeightCopier():
    def __init__(self,
                 bert_model: BertModel,
                 config: BertConfig
                 ) -> None:
        self.bert_model = bert_model
        self.config = config

    def build_layer(self) -> SimplifiedBertEmbeddings:
        """[summary]

        Returns:
            SimplifiedBertEmbeddings: Bert embedding layer with the
                same wights as the embedding layer of ´bert_model´.
        """

        bem = SimplifiedBertEmbeddings(self.config)

        bem.word_embeddings.weight = nn.Parameter(
            self.bert_model.embeddings.word_embeddings.weight)
        bem.token_type_embeddings.weight = nn.Parameter(
            self.bert_model.embeddings.token_type_embeddings.weight)
        bem.position_embeddings.weight = nn.Parameter(
            self.bert_model.embeddings.position_embeddings.weight)

        bem.LayerNorm.weight = nn.Parameter(
            self.bert_model.embeddings.LayerNorm.weight)
        bem.LayerNorm.bias = nn.Parameter(
            self.bert_model.embeddings.LayerNorm.bias)

        return bem


class BertClassifierCopier():
    def __init__(self,
                 bert_model: BertModel,
                 config: BertConfig
                 ) -> None:
        self.bert_model = bert_model
        self.config = config

    def build_layer(self) -> SimplifiedBertClassifier:
        """Initialize a SimplifiedBertClassifier and copy the weights
        of the Bert Model to the new model

        Returns:
            SimplifiedBertEncoder: Bert classifier with the same wights
                as ´bert_model´.
        """
        sbm = SimplifiedBertClassifier(self.config)
        sbm.embedding = BertEmbeddingsWeightCopier(
            self.bert_model.bert.embeddings, self.config).build_layer()
        sbm.encoder = BertEncoderWeightCopier(
            self.bert_model, self.config).build_layer()

        sbm.pooler.dense.weight = nn.Parameter(
            self.bert_model.bert.pooler.dense.weight)
        sbm.pooler.dense.bias = nn.Parameter(
            self.bert_model.bert.pooler.dense.bias)

        sbm.classifier.weight = nn.Parameter(
            self.bert_model.classifier.weight)
        sbm.classifier.bias = nn.Parameter(
            self.bert_model.classifier.bias)

        return sbm
