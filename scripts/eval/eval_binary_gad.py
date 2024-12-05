import torch
import json
import os
import sys
sys.path.append(".")
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor

SPLIT = "binary"
id = "binary_len_5_0"

NUM_ITER = 500
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

    # Initialize logits processor for the grammar
    gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, save_log=True)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        gad_oracle_processor,
    ])

    # Tokenize prompt into ids
    prompt = "Be a helpful assistant. Generate a random binary string of length 5? Directly show the generated string without explanation."
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    # Inference Loop
    history = []
    for _ in tqdm(range(NUM_ITER), desc="Running Inference"):
        # Generate sequences
        output = model.generate(
            input_ids,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=MAX_NEW_TOKENS,
            top_p=TOP_P,
            top_k=TOP_K,
            temperature=TEMPERATURE,
            logits_processor=logits_processors,
            repetition_penalty=REPETITION_PENALTY,
            num_return_sequences=1,
            return_dict_in_generate=True,
            output_scores=True,
        )

        input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
        # print(input_length, output.sequences[0])
        generated_tokens = output.sequences[0, input_length:].tolist()
        raw_likelihood = gad_oracle_processor.oracle_trie.raw_likelihood(generated_tokens)
        h = {"tokens" : generated_tokens, "raw_likelihood" : raw_likelihood}
        
        # Save history
        history.append(h)

        # Incremental parser state must be reset after each generation
        gad_oracle_processor.reset()

    # Save the history as JSON
    make_dir(f"{RESULT_PATH}/{id}")
    with open(f"{RESULT_PATH}/{id}/{id}_gad.jsonl", "w") as f:
        for h in history:
            f.write(json.dumps(h))
            f.write("\n")

if __name__=="__main__":
    main()