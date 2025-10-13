"""
TinyLlama model implementation for BloomBee distributed inference.
"""

import torch
from hivemind import get_logger

from bloombee.client.remote_sequential import RemoteSequential
from bloombee.models.tinyllama.block import WrappedLlamaBlock
from bloombee.models.tinyllama.config import DistributedLlamaConfig

logger = get_logger(__name__)


class DistributedLlamaModel(torch.nn.Module):
    def __init__(self, config: DistributedLlamaConfig, dht=None, **kwargs):
        super().__init__()
        self.config = config
        self.layers = RemoteSequential(config, dht=dht, **kwargs)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return self.layers(input_ids, attention_mask=attention_mask, **kwargs)


class DistributedLlamaForCausalLM(torch.nn.Module):
    def __init__(self, config: DistributedLlamaConfig, dht=None, **kwargs):
        super().__init__()
        self.config = config
        self.model = DistributedLlamaModel(config, dht=dht, **kwargs)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_ids, **kwargs):
        # Simple generation implementation
        with torch.no_grad():
            outputs = self.forward(input_ids, **kwargs)
            return outputs.argmax(dim=-1)


class DistributedLlamaForSequenceClassification(torch.nn.Module):
    def __init__(self, config: DistributedLlamaConfig, dht=None, **kwargs):
        super().__init__()
        self.config = config
        self.model = DistributedLlamaModel(config, dht=dht, **kwargs)
        self.score = torch.nn.Linear(config.hidden_size, config.num_labels, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.score(hidden_states)
        return logits


class DistributedLlamaForSpeculativeGeneration(torch.nn.Module):
    def __init__(self, config: DistributedLlamaConfig, dht=None, **kwargs):
        super().__init__()
        self.config = config
        self.model = DistributedLlamaModel(config, dht=dht, **kwargs)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask, **kwargs)
        logits = self.lm_head(hidden_states)
        return logits