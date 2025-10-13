from bloombee.models.tinyllama.block import WrappedLlamaBlock
from bloombee.models.tinyllama.config import DistributedLlamaConfig
from bloombee.models.tinyllama.model import (
    DistributedLlamaForCausalLM,
    DistributedLlamaForSequenceClassification,
    DistributedLlamaModel,
)
from bloombee.models.tinyllama.speculative_model import DistributedLlamaForSpeculativeGeneration
from bloombee.utils.auto_config import register_model_classes

register_model_classes(
    config=DistributedLlamaConfig,
    model=DistributedLlamaModel,
    model_for_causal_lm=DistributedLlamaForCausalLM,
    model_for_speculative=DistributedLlamaForSpeculativeGeneration,
    model_for_sequence_classification=DistributedLlamaForSequenceClassification,
)
