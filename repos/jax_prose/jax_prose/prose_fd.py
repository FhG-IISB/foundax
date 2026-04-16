from typing import Optional

import flax.linen as nn
import jax.numpy as jnp

from .config import PROSE1to1Config
from .embedder import get_embedder
from .transformer import TransformerDataEncoder, DataOperatorDecoder


class PROSE1to1(nn.Module):
    config: PROSE1to1Config
    x_num: int
    max_output_dim: int
    output_len: int = 1

    def setup(self):
        self.embedder = get_embedder(
            self.config.embedder, self.x_num, self.max_output_dim
        )
        self.data_encoder = TransformerDataEncoder(self.config.data_encoder)
        self.data_decoder = DataOperatorDecoder(
            self.config.data_decoder, output_len=self.output_len
        )

    def __call__(
        self,
        data_input: jnp.ndarray,
        input_times: jnp.ndarray,
        output_times: jnp.ndarray,
        deterministic: bool = True,
    ) -> jnp.ndarray:
        """
        Inputs:
            data_input:   (bs, input_len, x_num, x_num, data_dim)
            input_times:  (bs/1, input_len, 1)
            output_times: (bs/1, output_len, 1)
        Output:
            data_output:  (bs, output_len, x_num, x_num, data_dim)
        """
        bs = data_input.shape[0]
        data_tokens = self.embedder.encode(data_input, input_times)
        data_encoded = self.data_encoder(data_tokens, deterministic=deterministic)

        query_emb = self.data_decoder.get_query_emb(output_times)
        if query_emb.shape[0] == 1 and bs > 1:
            query_emb = jnp.broadcast_to(
                query_emb, (bs, query_emb.shape[1], query_emb.shape[2])
            )

        decoded = self.data_decoder(
            src=data_encoded,
            query_emb=query_emb,
            src_key_padding_mask=None,
            deterministic=deterministic,
        )
        return self.embedder.decode(decoded)
