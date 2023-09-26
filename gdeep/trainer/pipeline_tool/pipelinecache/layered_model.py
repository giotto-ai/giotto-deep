import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributed.pipeline.sync.skip import stash, pop, skippable 

@skippable(stash=['input_1_to_model_embedding_layer_0'])
class input_1_layer(nn.Module):
    def forward(self, input):
        ret = input
        yield stash('input_1_to_model_embedding_layer_0', ret)
        return input

@skippable(stash=['attention_mask_to_model_persformer_blocks_0_attention_layer_scaled_dot_product_attention', 'attention_mask_to_model_persformer_blocks_1_attention_layer_scaled_dot_product_attention', 'attention_mask_to_model_pooling_layer_scaled_dot_product_attention'])
class attention_mask_layer(nn.Module):
    def forward(self, input):
        ret = None
        yield stash('attention_mask_to_model_persformer_blocks_0_attention_layer_scaled_dot_product_attention', ret)
        yield stash('attention_mask_to_model_persformer_blocks_1_attention_layer_scaled_dot_product_attention', ret)
        yield stash('attention_mask_to_model_pooling_layer_scaled_dot_product_attention', ret)
        return input

@skippable(pop=['input_1_to_model_embedding_layer_0'])
class model_embedding_layer_0_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=4, out_features=16, bias=True)
    def forward(self, input):
        input_1 = yield pop('input_1_to_model_embedding_layer_0')
        ret = self.fc(input_1)
        return ret

@skippable(stash=['model_embedding_layer_1_to_add'])
class model_embedding_layer_1_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.GELU(approximate='none')
    def forward(self, input):
        ret = self.fc(input)
        yield stash('model_embedding_layer_1_to_add', ret)
        return ret

@skippable(pop=['attention_mask_to_model_persformer_blocks_0_attention_layer_scaled_dot_product_attention'])
class model_persformer_blocks_0_attention_layer_scaled_dot_product_attention_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.MultiheadAttention(embed_dim=16, num_heads=8, dropout=0.1, batch_first=True)
    def forward(self, input):
        attention_mask = yield pop('attention_mask_to_model_persformer_blocks_0_attention_layer_scaled_dot_product_attention')
        ret = self.fc(input, input, input, key_padding_mask=attention_mask)[0]
        return ret

class model_persformer_blocks_0_attention_layer_dropout_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Dropout(p=0.1, inplace=False)
    def forward(self, input):
        ret = self.fc(input)
        return ret

@skippable(stash=['add_to_add_1'], pop=['model_embedding_layer_1_to_add'])
class add_layer(nn.Module):
    def forward(self, input):
        model_embedding_layer_1 = yield pop('model_embedding_layer_1_to_add')
        ret = torch.add(input, model_embedding_layer_1)
        yield stash('add_to_add_1', ret)
        return ret

class model_persformer_blocks_0_feed_forward_layer_intermediate_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=16, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_persformer_blocks_0_feed_forward_layer_activation_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.GELU(approximate='none')
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_persformer_blocks_0_feed_forward_layer_dropout_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Dropout(p=0.1, inplace=False)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_persformer_blocks_0_feed_forward_layer_output_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=16, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

@skippable(stash=['add_1_to_add_2'], pop=['add_to_add_1'])
class add_1_layer(nn.Module):
    def forward(self, input):
        add = yield pop('add_to_add_1')
        input = input.clone()
        ret = torch.add(input, add)
        yield stash('add_1_to_add_2', ret)
        return ret

@skippable(pop=['attention_mask_to_model_persformer_blocks_1_attention_layer_scaled_dot_product_attention'])
class model_persformer_blocks_1_attention_layer_scaled_dot_product_attention_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.MultiheadAttention(embed_dim=16, num_heads=8, dropout=0.1, batch_first=True)
    def forward(self, input):
        attention_mask = yield pop('attention_mask_to_model_persformer_blocks_1_attention_layer_scaled_dot_product_attention')
        ret = self.fc(input, input, input, key_padding_mask=attention_mask)[0]
        return ret

class model_persformer_blocks_1_attention_layer_dropout_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Dropout(p=0.1, inplace=False)
    def forward(self, input):
        ret = self.fc(input)
        return ret

@skippable(stash=['add_2_to_add_3'], pop=['add_1_to_add_2'])
class add_2_layer(nn.Module):
    def forward(self, input):
        add_1 = yield pop('add_1_to_add_2')
        ret = torch.add(input, add_1)
        yield stash('add_2_to_add_3', ret)
        return ret

class model_persformer_blocks_1_feed_forward_layer_intermediate_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=16, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_persformer_blocks_1_feed_forward_layer_activation_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.GELU(approximate='none')
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_persformer_blocks_1_feed_forward_layer_dropout_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Dropout(p=0.1, inplace=False)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_persformer_blocks_1_feed_forward_layer_output_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=16, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

@skippable(stash=['add_3_to_expand', 'add_3_to_model_pooling_layer_scaled_dot_product_attention'], pop=['add_2_to_add_3'])
class add_3_layer(nn.Module):
    def forward(self, input):
        add_2 = yield pop('add_2_to_add_3')
        ret = torch.add(input, add_2)
        yield stash('add_3_to_expand', ret)
        yield stash('add_3_to_model_pooling_layer_scaled_dot_product_attention', ret)
        return ret

class model_pooling_layer_query_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.parameter.Parameter(torch.Tensor(1, 16), requires_grad=True)
    def forward(self, input):
        ret = self.fc
        return ret

@skippable(pop=['add_3_to_expand'])
class expand_layer(nn.Module):
    def forward(self, input):
        add_3 = yield pop('add_3_to_expand')
        ret = input.expand(add_3.shape[0], -1, -1)
        return ret

@skippable(pop=['add_3_to_model_pooling_layer_scaled_dot_product_attention', 'attention_mask_to_model_pooling_layer_scaled_dot_product_attention'])
class model_pooling_layer_scaled_dot_product_attention_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.MultiheadAttention(embed_dim=16, num_heads=8, dropout=0.1, batch_first=True)
    def forward(self, input):
        add_3 = yield pop('add_3_to_model_pooling_layer_scaled_dot_product_attention')
        attention_mask = yield pop('attention_mask_to_model_pooling_layer_scaled_dot_product_attention')
        ret = self.fc(input, add_3, add_3, key_padding_mask=attention_mask)[0]
        return ret

class squeeze_layer(nn.Module):
    def forward(self, input):
        ret = input.squeeze(dim=1)
        return ret

class model_classifier_layer_0_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=16, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_classifier_layer_1_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.GELU(approximate='none')
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_classifier_layer_2_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Dropout(p=0.1, inplace=False)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class model_classifier_layer_3_layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features=16, out_features=5, bias=True)
    def forward(self, input):
        ret = self.fc(input)
        return ret

class output_layer(nn.Module):
    def forward(self, input):
        ret = input
        return ret

class PipelinedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.s0 = nn.Sequential(input_1_layer(), attention_mask_layer(), model_embedding_layer_0_layer(), model_embedding_layer_1_layer(), model_persformer_blocks_0_attention_layer_scaled_dot_product_attention_layer(), model_persformer_blocks_0_attention_layer_dropout_layer(), add_layer(), model_persformer_blocks_0_feed_forward_layer_intermediate_layer(), model_persformer_blocks_0_feed_forward_layer_activation_layer(), model_persformer_blocks_0_feed_forward_layer_dropout_layer(), model_persformer_blocks_0_feed_forward_layer_output_layer(), add_1_layer()).cuda(0)
        self.s1 = nn.Sequential(model_persformer_blocks_1_attention_layer_scaled_dot_product_attention_layer(), model_persformer_blocks_1_attention_layer_dropout_layer(), add_2_layer(), model_persformer_blocks_1_feed_forward_layer_intermediate_layer(), model_persformer_blocks_1_feed_forward_layer_activation_layer(), model_persformer_blocks_1_feed_forward_layer_dropout_layer(), model_persformer_blocks_1_feed_forward_layer_output_layer(), add_3_layer(), model_pooling_layer_query_layer(), expand_layer(), model_pooling_layer_scaled_dot_product_attention_layer(), squeeze_layer(), model_classifier_layer_0_layer(), model_classifier_layer_1_layer(), model_classifier_layer_2_layer(), model_classifier_layer_3_layer(), output_layer()).cuda(1)
    def forward(self, input):
        ret = input
        ret = self.s0(ret.to(0))
        ret = self.s1(ret.to(1))
        return ret
    def get_modules(self):
        return  nn.Sequential(*[nn.Sequential(*self.s0),nn.Sequential(*self.s1)])
