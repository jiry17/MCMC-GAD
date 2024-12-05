import torch
import json
import os
import sys
sys.path.append(".")
import debug
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor

DATASET = "ebmoon/GAD-dataset"
# SPLIT = "SLIA"
SPLIT = "BV4"
# SPLIT = "CP"

NUM_ITER = 2000
#MODEL_ID = "mistralai/mistral-7b-instruct-v0.2"
MODEL_ID = "TinyLlama/TinyLlama_v1.1"
TRIE_PATH = f"tries/{SPLIT}"
RESULT_PATH = f"results/{SPLIT}"
DEVICE = "cuda"
# DEVICE = "cpu"
DTYPE = torch.bfloat16
MAX_NEW_TOKENS = 128
TEMPERATURE = 1.0
REPETITION_PENALTY = 1.0
TOP_P = 1.0
TOP_K = 0

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def eval_prob(model, tokenizer, id, prompt, grammar_str):
    # Load EBNF grammar
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Initialize logits processor for the grammar
    gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, save_log=True)
    inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
    logits_processors = LogitsProcessorList([
        inf_nan_remove_processor,
        gad_oracle_processor,
    ])

    # Tokenize prompt into ids
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    # Inference Loop
    outputs = []
    history = []
    for _ in tqdm(range(NUM_ITER), desc="Running Inference"):
        # Generate sequences
        # print("\nnew sample")
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
        generated_tokens = output.sequences[0, input_length:].tolist()
        raw_likelihood = gad_oracle_processor.oracle_trie.raw_likelihood(generated_tokens)
        h = {"tokens" : generated_tokens, "raw_likelihood" : raw_likelihood}
        # print(raw_likelihood)

        #if debug.is_target_list(generated_tokens):
        #    print("found target")
        #    exit(0)
        
        # Save history
        history.append(h)

        # Incremental parser state must be reset after each generation
        gad_oracle_processor.reset()

    # Save the history as JSON
    make_dir(f"{RESULT_PATH}/{id}")
    print(f"save to {RESULT_PATH}/{id}/{id}_test_gad.jsonl")
    with open(f"{RESULT_PATH}/{id}/{id}_test_gad.jsonl", "w") as f:
        for h in history:
            f.write(json.dumps(h))
            f.write("\n")
            
def main():
    dataset = load_dataset(DATASET, split=SPLIT)
    for data in dataset:
        id = data['id']
        prompt = data['prompt']
        grammar = data['grammar']

        if id != sys.argv[1]: continue

        device = torch.device(DEVICE)
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        tokenizer.pad_token = tokenizer.eos_token

        # Load model
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
        model.to(device)
        model.to(dtype=DTYPE)
        model.resize_token_embeddings(len(tokenizer))

        eval_prob(model, tokenizer, id, prompt, grammar)

        print(f"Evaluation finished: {id}")


if __name__=="__main__":
    print("Running for", sys.argv[1])
    main()