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

from transformers_gad.generation.mcmc_process import BeamMCMC, RawMCMCHolder
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor

DATASET = "ebmoon/GAD-dataset"
# SPLIT = "SLIA"
SPLIT = "BV4"
# SPLIT = "CP"

NUM_ITER = 100
# MODEL_ID = "mistralai/mistral-7b-instruct-v0.2"
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
MCMC_ITER_NUM = 200


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def eval_prob(model, tokenizer, id, prompt, grammar_str):
    # Load EBNF grammar
    grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

    # Tokenize prompt into ids
    input_ids = tokenizer(
        [prompt], add_special_tokens=False, return_tensors="pt", padding=True
    )["input_ids"]
    input_ids = input_ids.to(model.device)

    # Inference Loop
    outputs = []
    history = []

    def generator(processor):
        inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
        logits_processors = LogitsProcessorList([
            inf_nan_remove_processor,
            processor,
        ])
        return model.generate(
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
            # return_dict_in_generate=True,
            # output_scores=True,
        )[0]

    holder = RawMCMCHolder(input_ids[0], generator, tokenizer, grammar, DEVICE)

    for _ in tqdm(range(NUM_ITER), desc="Running Inference"):
        # res = holder.mcmc(MCMC_ITER_NUM)
        res = holder.mcmc(0)
        generated_tokens = res.prefix[input_ids.shape[1]:].cpu().tolist()
        if debug.is_target_list(generated_tokens):
            print("found target", debug.PROB_TRUTH)
            exit(0)
        # h = {"tokens": res.prefix[input_ids.shape[1]:].cpu().tolist(), "raw_likelihood": res.prefix_prob["raw"]}

        # Save history
        # history.append(h)
        # print(sum(holder.history) / len(holder.history))
    exit(0)
    # Save the history as JSON
    make_dir(f"{RESULT_PATH}/{id}")
    with open(f"{RESULT_PATH}/{id}/{id}_raw_mcmc.jsonl", "w") as f:
        json.dump(holder.history, f, indent=4)


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

    dataset = load_dataset(DATASET, split=SPLIT)
    for data in dataset:
        id = data['id']
        prompt = data['prompt']
        grammar = data['grammar']
        #print("New task", id)
        #print(prompt)
        if id != debug.TARGET_TASK: continue

        eval_prob(model, tokenizer, id, prompt, grammar)

        print(f"Evaluation finished: {id}")

if __name__ == "__main__":
    main()