import os
from typing import Optional, Union

from hivemind import get_logger
from transformers.models.llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from bloombee.client.config import ClientConfig
from bloombee.client.lm_head import LMHeadConfig
from bloombee.client.ptune import PTuneConfig
from bloombee.models.tinyllama.block import WrappedLlamaBlock

logger = get_logger(__name__)


class DistributedLlamaConfig(LlamaConfig, ClientConfig, PTuneConfig, LMHeadConfig):
    model_type = "tinyllama"
    block_class = WrappedLlamaBlock
    attn_class = LlamaAttention
    block_prefix = "model.layers"

    @property
    def num_key_value_groups(self):
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: Union[str, os.PathLike, None], *args, dht_prefix: Optional[str] = None, **kwargs
    ):
        logger.info(
            "Make sure you follow the Llama terms of use: "
            "https://llama.meta.com/llama3/license, https://llama.meta.com/llama2/license"
        )

        loading_from_repo = model_name_or_path is not None and not os.path.isdir(model_name_or_path)
        if loading_from_repo and dht_prefix is None:
            dht_prefix = str(model_name_or_path)
            dht_prefix = dht_prefix.split("/")[-1]  # Use only repo name to merge blocks hosted by different accounts
            dht_prefix = dht_prefix.replace(".", "-")
            if not dht_prefix.endswith("-hf"):
                dht_prefix += "-hf"
            logger.info(f"Using DHT prefix: {dht_prefix}")

        result = super().from_pretrained(model_name_or_path, *args, dht_prefix=dht_prefix, **kwargs)
        config = result[0] if isinstance(result, tuple) else result
        
        # Force TinyLlama models to use our TinyLlama implementation
        if "tinyllama" in str(model_name_or_path).lower():
            original_model_type = config.model_type
            config.model_type = "tinyllama"
            logger.info(f"Overriding model_type from '{original_model_type}' to 'tinyllama' for {model_name_or_path}")
            logger.info(f"Config model_type is now: {config.model_type}")
            
            # Override TinyLlama-specific dimensions
            config.hidden_size = 2048
            config.num_hidden_layers = 22
            config.num_attention_heads = 32
            config.num_key_value_heads = 4
            config.intermediate_size = 5632
            config.vocab_size = 32000
            config.max_position_embeddings = 2048
            logger.info(f"Set TinyLlama dimensions: hidden_size={config.hidden_size}, num_hidden_layers={config.num_hidden_layers}, num_attention_heads={config.num_attention_heads}")
            
            # Force the block_class to use TinyLlama block class
            from bloombee.models.tinyllama.block import WrappedLlamaBlock
            config.block_class = WrappedLlamaBlock
            logger.info(f"Set block_class to: {config.block_class}")
        
        config.pretraining_tp = 1  # This may give less accurate results but it doesn't matter if we use quantization
        config.use_cache = True  # use_cache=False leads to identical results but is slower and not supported by Petals
        return result
