import copy
import math
import torch.nn.functional as F

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.oracle.oracle_trie import Trie
from transformers_gad.recognizer import AcceptState

RAW = "raw"
MASKED = "masked"

class SampleNode:
    def __init__(self, prefix, state: AcceptState):
        self.prefix = prefix
        self.state = state
        self.prefix_prob = None
        self.next_prob = None

    def is_new(self):
        return self.prefix_prob is None

    def to_string(self, tokenizer):
        return tokenizer.decode(self.prefix[0])

def _process_score(scores):
    scores[scores != scores] = 0.0

    scores[scores == float("inf")] = torch.finfo(scores.dtype).max
    scores[scores == float("-inf")] = torch.finfo(scores.dtype).min
    # return F.softmax(scores, dim=-1)

def draw_sample(model, tokenizer, node: SampleNode, constraint: IncrementalGrammarConstraint, DEVICE):
    while True:
        if node.is_new():
            logits = model(node.prefix).logits
            raw_score = logits[0, -1, :]
            _process_score(raw_score)
            vocab = constraint.filter_vocab(node.state, DEVICE)
            print(vocab.nonzero())

            raw_prob = _process_score(raw_score)





    full_text = prompt + " " + s
    tokens = tokenizer.encode(full_text, return_tensors='pt')
    start_index = len(tokenizer.encode(prompt + " ", return_tensors="pt")[0]) - 1
    print(start_index)
    llm_prob = 1.0
    for i in range(start_index, len(tokens[0]) - 1):
        input_tokens = tokens[:, :i + 1].to(DEVICE)
        outputs = model(input_tokens)
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        probabilities = F.softmax(last_token_logits, dim=-1)

        next_token_id = tokens[0, i + 1]
        next_token_prob = probabilities[next_token_id].item()
        context = tokenizer.decode(input_tokens[0])
        next_token = tokenizer.decode([next_token_id])
