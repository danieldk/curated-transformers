from dataclasses import dataclass

import torch

from ...layers.activations import Activation
from ..config import (
    RotaryEmbeddingConfig,
    TransformerAttentionLayerConfig,
    TransformerConfig,
    TransformerEmbeddingLayerConfig,
    TransformerFeedForwardLayerConfig,
    TransformerLayerConfig,
)


@dataclass
class MistralConfig(TransformerConfig):
    """
    Mistral model configuration.
    """

    def __init__(
        self,
        *,
        activation: Activation = Activation.SiLU,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        hidden_width: int = 4096,
        intermediate_width: int = 14336,
        layer_norm_eps: float = 1e-5,
        model_max_length: int = 32768,
        n_hidden_layers: int = 32,
        n_key_value_heads: int = 8,
        n_pieces: int = 50280,
        n_positions: int = 32768,
        n_query_heads: int = 32,
        rotary_embedding_base: int = 10000,
        rotary_embedding_fraction: float = 0.25,
        sliding_window: int = 4096
    ):
        """
        :param activation:
            Activation used by the pointwise feed-forward layers.
        :param attention_probs_dropout_prob:
            Dropout to apply after attention.
        :param hidden_dropout_prob:
            Dropout to apply to the hidden and embedding layers.
        :param hidden_width:
            Hidden width of the transformer.
        :param intermediate_width:
            Intermediate width in the feed-forward layer. The non-linearity
            is applied in this intermediate width.
        :param layer_norm_eps:
            Epsilon for layer normalization.
        :param n_hidden_layers:
            Number of hidden layers.
        :param n_key_value_heads:
            Number of key-value heads.
        :param n_pieces:
            Vocabulary size (number of embeddings).
        :param n_query_heads:
            Number of query heads.
        :param rotary_embedding_base:
            Base in signifying the rotary embedding period.
        :param rotary_embedding_fraction:
            Fraction of hidden width to apply rotary embeddings to.
            Must be in ``[0,1]``.
        """

        # TODO: n_positions and model_max_length are currently
        #       not used. We may want to limit the rotary embeddings to these
        #       values in the future. We should check empirically if the auto
        #       resizing in rotary embeddings makes sense.

        self.embedding = TransformerEmbeddingLayerConfig(
            dropout_prob=hidden_dropout_prob,
            embedding_width=hidden_width,
            n_pieces=n_pieces,
            layer_norm_eps=layer_norm_eps,
            n_positions=None,
            n_types=None,
        )
        self.layer = TransformerLayerConfig(
            attention=TransformerAttentionLayerConfig(
                dropout_prob=attention_probs_dropout_prob,
                hidden_width=hidden_width,
                n_query_heads=n_query_heads,
                n_key_value_heads=n_key_value_heads,
                rotary_embeddings=RotaryEmbeddingConfig(
                    rotary_fraction=rotary_embedding_fraction,
                    rotary_base=rotary_embedding_base,
                ),
                use_alibi=False,
                use_bias=False,
                use_parallel_attention=False,
            ),
            feedforward=TransformerFeedForwardLayerConfig(
                hidden_width=hidden_width,
                intermediate_width=intermediate_width,
                activation=activation,
                use_bias=False,
                use_gate=False,
            ),
            dropout_prob=hidden_dropout_prob,
            layer_norm_eps=layer_norm_eps,
            n_hidden_layers=n_hidden_layers,
        )
        self.dtype = torch.float16
