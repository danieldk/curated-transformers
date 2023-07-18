from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.nn import Identity, Module

from .attention import AttentionMask, KeyValueCache, SelfAttention
from .feedforward import PointwiseFeedForward


@dataclass
class TransformerLayerNorms:
    """
    Layer normalizations used in a transformer layer.

    By default, all the normalizations are disabled by setting the layer
    normalization to the Torch ``Identity`` module. Therefore, only
    normalizations that are needed have to be set.
    """

    #: Normalization of the input to the attention layer.
    attn_input_layer_norm: Module = Identity()

    #: Normalization of the output of the attention layer after the
    #: residual connection.
    attn_residual_layer_norm: Module = Identity()

    #: Normalization of the input to the feed-forward layer.
    ffn_input_layer_norm: Module = Identity()

    #: Normalization of the output of the feed-forward layer after the
    #: residual connection.
    ffn_residual_layer_norm: Module = Identity()


class _TransformerLayer(Module):
    """
    Transformer decoder layer (`Vaswani et al., 2017`_) base class.

    This is a generic transformer layer. :py:class:`DecoderLayer` and
    :py:class:`EncoderLayer` provide specialized encoder/decoder layers.

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    # Don't expose this module, we cannot provide the full forward method
    # without violating the Liskov substitution principle. Ideally, we'd
    # use this module using composition inside the encoder/decoder layers,
    # but we can't without adding a prefix.

    def __init__(
        self,
        *,
        attention_layer: SelfAttention,
        feed_forward_layer: PointwiseFeedForward,
        hidden_dropout: float,
        layer_norms: TransformerLayerNorms,
        parallel_attention: bool,
    ):
        """
        Construct a transformer layer.

        :param attention_layer:
            The attention layer to use in the transformer layer.
        :param feed_forward_layer:
            The pointwise feed-forward layer to use in the transformer layer.
        :param hidden_dropout:
            Dropout probabilty to apply after hidden layers.
        :param layer_norms:
            Layer norms to use in the layer.
        :param parallel_attention:
             Use parallel attention.
        """
        super().__init__()

        self.parallel_attention = parallel_attention

        self.mha = attention_layer
        self.attn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

        self.ffn = feed_forward_layer
        self.ffn_output_dropout = torch.nn.Dropout(p=hidden_dropout)

        self.attn_input_layer_norm = layer_norms.attn_input_layer_norm
        self.attn_residual_layer_norm = layer_norms.attn_residual_layer_norm
        self.ffn_input_layer_norm = layer_norms.ffn_input_layer_norm
        self.ffn_residual_layer_norm = layer_norms.ffn_residual_layer_norm

    def _forward(
        self,
        input: Tensor,
        *,
        use_causal_mask: bool,
        attention_mask: Optional[AttentionMask],
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the transformer layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :param use_causal_mask:
            Mask out succeeding sequence elements when ``True``.
        :param cache:
            Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        residual = input

        attn_out, cache = self.mha(
            self.attn_input_layer_norm(input),
            attention_mask,
            cache=cache,
            store_cache=store_cache,
            positions=positions,
            use_causal_mask=use_causal_mask,
        )
        attn_out = self.attn_output_dropout(attn_out)

        if self.parallel_attention:
            ffn_in = input
        else:
            residual = self.attn_residual_layer_norm(input + attn_out)
            ffn_in = residual

        ffn_out = self.ffn(self.ffn_input_layer_norm(ffn_in))
        ffn_out = self.ffn_output_dropout(ffn_out)

        if self.parallel_attention:
            output = attn_out + ffn_out
        else:
            output = ffn_out

        return self.ffn_residual_layer_norm(residual + output), cache


class DecoderLayer(_TransformerLayer):
    """
    Transformer decoder layer (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[AttentionMask],
        cache: Optional[KeyValueCache] = None,
        positions: Optional[Tensor] = None,
        store_cache: bool = False,
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the decoder layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :param cache:
            Key/value cache to avoid recomputing
            key/value representations for tokens that were previously seen.
        :param positions:
            Input positions. Positions are needed to look up rotary embeddings.
            Normally, these positions are calculated automatically. But if the
            positions deviate for some reason, they can be provided through this argument.
        :param store_cache:
            Whether to cache the key/value representations for future reuse.
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return super()._forward(
            input,
            attention_mask=attention_mask,
            cache=cache,
            use_causal_mask=True,
            positions=positions,
            store_cache=store_cache,
        )


class EncoderLayer(_TransformerLayer):
    """
    Transformer encoder layer (`Vaswani et al., 2017`_).

    .. _Vaswani et al., 2017: https://arxiv.org/abs/1706.03762
    """

    def forward(
        self,
        input: Tensor,
        attention_mask: Optional[AttentionMask],
    ) -> Tuple[Tensor, Optional[KeyValueCache]]:
        """
        Apply the encoder layer to the given piece hidden representations.

        :param input:
            Hidden representations to apply the layer to.

            *Shape:* ``(batch_size, seq_len, width)``
        :param attention_mask:
            Attention mask. Sequence elements for which the
            corresponding mask element is set to ``False`` are ignored
            during attention calculation.

            *Shape:* ``(batch_size, seq_len)``
        :returns:
            Layer output and the key/value cache.

            *Shape:* ``(batch_size, seq_len, width)``
        """
        return super()._forward(
            input, attention_mask=attention_mask, use_causal_mask=False
        )