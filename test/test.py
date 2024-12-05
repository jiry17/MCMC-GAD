import torch
import json
import os
import sys

sys.path.append(".")
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor, \
    GrammarConstrainedLogitsProcessor

SPLIT = "binary"
id = "binary_len_5_0"

NUM_ITER = 100
MODEL_ID = "TinyLlama/TinyLlama_v1.1"
# MODEL_ID = "mistralai/mistral-7b-instruct-v0.2"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
TRIE_PATH = f"tries/{SPLIT}"
RESULT_PATH = f"results/{SPLIT}"
DEVICE = "cuda"
# DEVICE = "cuda"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 512
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

import random

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def main():
    device = torch.device(DEVICE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.to(device)
    model.to(dtype=DTYPE)
    model.resize_token_embeddings(len(tokenizer))

    # Load EBNF grammar
    with open(GRAMMAR_PATH, "r") as file:
        grammar_str = file.read()
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Tokenize prompt into ids
    prompt = "Be a helpful assistant. Generate a random binary string of length 5? Directly show the generated string without explanation."
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    # Initialize logits processor for the grammar
    gcd_oracle_processor = GrammarConstrainedLogitsProcessor(grammar, parse_start_index=input_ids.shape[1])
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        gcd_oracle_processor,
    ])

    # Inference Loop
    history = []
    for _ in range(100):
        prefix_size = random.randint(0, 5)
        while True:
            prefix = [random.choice([29896, 29900]) for _ in range(prefix_size)]
            if len(prefix) == 0:
                break
            if prefix[0] == 29900 and min(prefix) != prefix[0]: continue
            break


        gcd_oracle_processor.set_prefix_restriction(prefix)

        # Generate sequences
        output = model.generate(
            input_ids,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS - prefix_size,
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            logits_processor=logits_processors,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )


if __name__ == "__main__":
    main()