import torch
import torch.nn as nn

# Huggingface Bert implementation
from transformers import BertConfig

# extended einstein operations for tensor operations
from einops import rearrange


class SimplifiedMultiHeadSelfAttention(nn.Module):
    """
    Simplified multi head self-attention layer with einops wherever possible

    Args:
        config (BertConfig): Bert model specification
    """
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        self.config = config

        # dimension of the encoder and pooler layer
        self.dim = config.hidden_size
        self.head = config.num_attention_heads  # number of attention heads
        # dimension of each attention head
        self.dim_head = int(self.dim / self.heads)
        # linear layer splitting hidden state to query, key, value
        self.to_qvk = nn.Linear(self.dim, 3 * self.dim, bias=True)
        # scaling factor in row-wise softmax to prevent small gradients
        self.scale_factor = self.dim_head ** -0.5

    def forward(self, x: torch.tensor, mask: torch.tensor = None
                ) -> torch.tensor:
        """ Forward pass

        Args:
            x (torch.tensor): input tensor of shape [batch, tokens, dim]
            mask (torch.tensor, optional): attention mask of shape
                [batch, tokens, dim]. Defaults to None.

        Returns:
            torch.tensor: attention result concatenated head-wise
        """
        assert x.dim() == 3  # [batch, tokens, dim]
        qkv = self.to_qvk(x)  # [batch, tokens, 3 * heads * dim_heads]

        # decomposing to query, key, value each with dim [batch, heads, tokens,
        # dim_heads]
        q, k, v = tuple(rearrange(qkv, 'b t (qkv h d) -> qkv b h t d',
                                  qkv=3,
                                  h=self.head))

        # attention value of shape [batch, heads, tokens, dim_heads]
        out = self.__compute_mhsa(q, k, v, mask=mask)
        # merge heads
        out = rearrange(out, 'b h t d -> b t (h d)')
        return out

    def __compute_mhsa(
                self,
                q: torch.tensor,
                k: torch.tensor,
                v: torch.tensor,
                mask: torch.tensor = None) -> torch.tensor:
        """ Implemenatation of multi head self attention values

        Args:
            q (torch.tensor): query tensor of shape [batch, tokens, dim]
            k (torch.tensor): key tensor of shape [batch, tokens, dim]
            v (torch.tensor): value tensor of shape [batch, tokens, dim]
            mask (torch.tensor, optional): attention mask of shape.
                Defaults to None.

        Returns:
            torch.tensor: attention values of shape [batch, tokens, dim]
        """
        # scalar product of query and key vectors
        scaled_dot_prod = torch.einsum('... i d, ... j d -> ... i j', q, k)\
            * self.scale_factor

        if mask is not None:
            try:
                scaled_dot_prod = scaled_dot_prod.masked_fill(
                    mask == 0,
                    -float('inf')
                    )
            except Exception:
                print(
                    """"
                    mask and scaled dot product do not have the same
                    dimensionality!
                    """
                    )

        # Row-wise softmax
        attention = torch.softmax(scaled_dot_prod, dim=-1)
        return torch.einsum('... i j, ... j d -> ... i d', attention, v)


class SimplifiedBertBlock(nn.Module):
    """
    Bert transformer block consisting of a multi head attention layer
    and a feed-forward layer

    Args:
        config (BertConfig): Bert model specification
    """
    def __init__(self, config: BertConfig) -> None:
        super().__init__()
        # set activation function as nn.Module
        activation = getattr(nn, config.hidden_act.upper())()

        dim = config.hidden_size  # token's vector length
        # dimension of the inner linear layer
        intermediate_dim = config.intermediate_size
        heads = config.num_attention_heads  # number of attention heads
        # value added to the denominator of the normalization for numerical
        # stability
        layer_norm_eps = config.layer_norm_eps
        dropout = config.dropout  # probabilty of dropping values

        #  attention layer
        self.mhsa = SimplifiedMultiHeadSelfAttention(dim, heads)
        self.attention_linear = nn.Linear(dim, dim)
        self.drop_1 = nn.Dropout(dropout)
        self.norm_1 = nn.LayerNorm(dim, eps=layer_norm_eps)
        self.drop_2 = nn.Dropout(dropout)

        #  feed forwad layer, see https://arxiv.org/abs/1706.03762
        self.linear = nn.Sequential(
            nn.Linear(dim, intermediate_dim),
            activation,
            nn.Linear(intermediate_dim, dim),
            nn.Dropout(dropout)
        )
        self.norm_2 = nn.LayerNorm(dim, eps=layer_norm_eps)

    def forward(self, x: torch.tensor, mask: torch.tensor = None
                ) -> torch.tensor:
        attention_output = self.attention_linear(
            self.drop_1(self.mhsa(x, mask))
            )
        attention_normalized = self.drop_2(self.norm_1(attention_output + x))
        return self.norm_2(
            self.linear(attention_normalized) + attention_normalized
            )


class SimplifiedBertEncoder(nn.Module):
    """
    Bert encoder consisting of stacked Bert blocks

    Args:
        config (BertConfig): Bert model specification
    """
    def __init__(self, config: BertConfig) -> None:
        super().__init__()

        # number of transformer blocks
        blocks = config.num_hidden_layers

        # stacking ´blocks´ bert transformer blocks
        self.block_list = [
                        SimplifiedBertBlock(config)
                        for _ in range(blocks)
        ]
        # properly register ´block_list´ to make them visible to all ´Module´
        # methods
        self.layers = nn.ModuleList(self.block_list)

    def forward(self, x: torch.tensor, mask: torch.tensor = None
                ) -> torch.tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x


class SimplifiedBertEmbeddings(nn.Module):
    """
    Simplified Bert Embedding layer consisting of
    * Word embedding layer
    * token type embedding layer
    * (absolute) positional encoding layer with trainable weights
    Only works for absolute embeddings up to now.

    Get the embeddings from one-hot encodings of the words.

    Args:
    config (BertConfig): Bert model specification
    """
    def __init__(self, config: BertConfig):
        super().__init__()

        dim = config.hidden_size  # token's vector length

        # vocabulary size of Bert model, defaults to 30522
        vocab_size = config.vocab_size

        # Token id for padding, for example when batching sequences,
        # defaults to 0
        pad_token_id = config.pad_token_id

        # number of different token types, default to 2
        type_vocab_size = config.type_vocab_size

        # Maximal sequence length that can be passed to the model,
        # defaults to 512
        max_position_embeddings = config.max_position_embeddings

        # value added to the denominator of the normalization for numerical
        # stability
        layer_norm_eps = config.layer_norm_eps
        dropout = config.dropout  # probabilty of dropping values

        self.word_embeddings = nn.Embedding(vocab_size,
                                            dim,
                                            padding_idx=pad_token_id)
        self.token_type_embeddings = nn.Embedding(type_vocab_size,
                                                  dim)
        # absolute positional embedding
        self.position_embeddings = nn.Embedding(max_position_embeddings, dim)

        # position_ids that will be encoded by the ´position_embeddings´ layer
        # register_buffered are non-trainable weigths
        self.register_buffer(
            "position_ids",
            torch.arange(config.max_position_embeddings).expand((1, -1))
        )

        self.LayerNorm = nn.LayerNorm(dim, layer_norm_eps)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self,
                input_ids: torch.tensor,
                token_type_ids: torch.tensor
                ):
        seq_length = input_ids.shape[1]

        inputs_embeds = self.word_embeddings(input_ids)  # word embedding+

        # token embedding
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # shrink position_ids to seq_length
        position_ids = self.position_ids[:, : seq_length]
        absolute_embeddings = self.position_embeddings(position_ids)

        embeddings = inputs_embeds + token_type_embeddings\
            + absolute_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout_layer(embeddings)
        return embeddings


class SimplifiedBertPooler(nn.Module):
    """
    Bert Pooler (only for classifications tasks)
    Takes the output representation corresponding to the [CLS] token
    and uses it for downstream tasks like sentiment analysis or next
    sentence prediction. It applies a linear transformation over the
    [CLS] token.

    Args:
        config (BertConfig): Bert model specification
    """
    def __init__(self, config):
        super().__init__()
        dim = config.hidden_dim  # token's vector length

        self.dense = nn.Linear(dim, dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # extract classification token
        first_token_tensor = x[:, 0]
        # feed-forward layer
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class SimplifiedBertClassifier(nn.Module):
    """
    Bert classifier with ´num_labels´ of output classes consisting of
    * Embedding layer
    * Bert encoder consisting of stacked self attention layers
    * Pooler layer
    * Linear classifier layer computing the logit probabilities of the
      different classes

    Args:
        config (BertConfig): Bert model specification
    """
    def __init__(self, config: BertConfig, num_labels: int = 2) -> None:
        super().__init__()
        dim = config.hidden_size  # token's vector length

        self.embedding = SimplifiedBertEmbeddings(config)

        self.encoder = SimplifiedBertEncoder(config)

        self.pooler = SimplifiedBertPooler(config)

        self.classifier = nn.Linear(dim, num_labels)

    def forward(self,
                input_ids: torch.tensor,
                token_type_ids: torch.tensor,
                attention_mask: torch.tensor
                ) -> torch.tensor:
        x = self.embedding(input_ids, token_type_ids)

        x = self.encoder(x, attention_mask)

        x = self.pooler(x)

        return self.classifier(x)
