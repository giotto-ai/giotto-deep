"""
Testing for bert_weight_copier.
Hacked by Raphael Reinauer 2021 (https://github.com/raphaelreinauer)
"""

import torch

import gdeep.bert.bert_weight_copier

def test_embedding_layer_copier():
    bec = BertEmbeddingsWeightCopier(model.bert.embeddings, config)
    bem = bec.build_layer()
    bem.eval()

    assert torch.allclose(
        model.bert.embeddings.token_type_embeddings(inputs.token_type_ids),
        bem.token_type_embeddings(inputs.token_type_ids)
        )

    assert torch.allclose(
        bem(input_ids = inputs['input_ids'], token_type_ids=inputs['token_type_ids']),
        model.bert.embeddings(input_ids = inputs['input_ids'], token_type_ids=inputs['token_type_ids'])
    )
    
def test_bert_classifier_copier():
    bc = SimplifiedBertClassifier()
    bc.embedding = BertEmbeddingsWeightCopier(model.bert.embeddings, config).build_layer()
    bc.encoder = BertEncoderWeightCopier(model, config).build_layer()

    bc.pooler.dense.weight = nn.Parameter(model.bert.pooler.dense.weight)
    bc.pooler.dense.bias = nn.Parameter(model.bert.pooler.dense.bias)

    bc.classifier.weight = nn.Parameter(model.classifier.weight)
    bc.classifier.bias = nn.Parameter(model.classifier.bias)

    bc.eval()

    assert torch.allclose(
        model.bert.embeddings(input_ids = inputs['input_ids'], token_type_ids=inputs['token_type_ids']),
        bc.embedding(input_ids = inputs['input_ids'], token_type_ids=inputs['token_type_ids'])
    )

    for idx_layer in range(12 - 1):
        assert torch.allclose(
            bc.encoder.layers[idx_layer](hidden[idx_layer]),
            hidden[idx_layer + 1]
        )

    x = hidden[0]
    for layer_idx in range(12):
        x = bc.encoder.layers[layer_idx](x)
    assert torch.allclose(
        x,
        hidden[-1]
    )