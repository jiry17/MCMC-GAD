# Grammar-Aligned Decoding

## About

This repository implements the **A**daptive **S**ampling with **Ap**proximate expected futures (ASAp) algorithm, introduced in the paper [Grammar-Aligned Decoding](https://arxiv.org/abs/2405.21047). The ASAP algorithm addresses the issue with GCD techniques (and constrained decoding methods in general), which can distort the LLM's probability distribution. 

## Installation

Clone the repository:
```
git clone git@github.com:ebmoon/transformers-GAD.git
```
Create a new Conda environment using the provided requirements file. Replace `<env>` with the actual name of your environment:
```
conda create --name <env> python=3.11
conda activate <env>
pip install -r requirements.txt
```

## Examples

### Quick Start

```python
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation.logits_process import LogitsProcessorList, InfNanRemoveLogitsProcessor
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.generation.logits_process import GrammarAlignedOracleLogitsProcessor

MODEL_ID = "TinyLlama/TinyLlama_v1.1"
GRAMMAR_PATH = "examples/test/binary_len_5_0.ebnf"
MAX_NEW_TOKENS = 512

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID)

# Load EBNF grammar
with open(GRAMMAR_PATH, "r") as f:
    grammar_str = f.read()
grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)

# Initialize logits processor for the grammar
gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar)
inf_nan_remove_processor = InfNanRemoveLogitsProcessor()
logits_processors = LogitsProcessorList([
    inf_nan_remove_processor,
    gad_oracle_processor,
])

# Tokenize prompt into ids
prompt = "Generate a binary string of length 5"
input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")["input_ids"]

# Inference Loop
outputs = []
for _ in tqdm(range(10), desc="Running Inference"):
    # Generate sequences
    output = model.generate(
        input_ids,
        do_sample=True,
        max_new_tokens=MAX_NEW_TOKENS,
        logits_processor=logits_processors
    )

    # Incremental parser state must be reset after each generation
    gad_oracle_processor.reset()

    # Detokenize generated output
    input_length = 1 if model.config.is_encoder_decoder else input_ids.shape[1]
    generated_tokens = output.sequences[:, input_length:]
    generations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    outputs.append(generations[0])

print(outputs)
```

The ASAp algorithm is implemented as a logit processor. Users can initialize a new `GrammarAlignedOracleLogitsProcessor` for an EBNF grammar and pass it as an argument during generation. Since the logit processor uses an incremental parser internally, users must manually reset the parser state ahead of the next generation the generation.

You can try running `scripts/test_gad.py`. 

### Using Trained ASAp Trie

Trained ASAp tries can be saved as a JSON file.

```python
with open(TRIE_PATH, "w") as f:
    f.write(gad_oracle_processor.oracle_trie.json())
```

Saved ASAp tries can be reloaded from a previously saved JSON file and passed during the initialization of the`GrammarAlignedOracleLogitsProcessor`. The full example can be checked in `scripts/test_gad.py`.

```python
from transformers_gad.oracle.oracle_trie import Trie

with open(TRIE_PATH, "r") as f:
    trie = Trie.loads(f.read())

grammar = IncrementalGrammarConstraint(grammar_str, "root", tokenizer)
gad_oracle_processor = GrammarAlignedOracleLogitsProcessor(grammar, trie)
```

The full example can be checked in `scripts/test_gad_load_trie.py`.

## Evaluation


### Dataset and Checkpoints

* [Evaluation dataset](https://huggingface.co/datasets/ebmoon/GAD-dataset)
* Fine-tuning
    * [SLIA checkpoints](https://huggingface.co/MilaWang/Mistral-7B-Instruct-v0.2-gad-slianogram3-merged)
    * [BV4 checkpoints](https://huggingface.co/MilaWang/Mistral-7B-Instruct-v0.2-gad-bv4nogram3-merged)
    * [CP checkpoints](https://huggingface.co/MilaWang/Mistral-7B-Instruct-v0.2-gad-cp8-merged)

### Scripts

Running scripts in `scripts/eval` collects data required for plot. 

* `eval_binary_gad.py` and `eval_binary_gcd.py` collect data for the skewed binary grammar example.
* `eval_gad.py` and `eval_gcd.py` collect data for `SLIA`, `BV` and `CP` dataset. To specify which dataset to use, you must manually set the `SPLIT` variable to either `"SLIA"`, `"BV"` or `"CP"`.


Running scripts in `scripts/plot` will generate plots from collected data.

* Again, to specify which dataset to use, you must manually set the `SPLIT` variable to either `"binary"`, `"SLIA"`, `"BV"` or `"CP"`.
* `plots/{SPLIT}/prob` contains plots for expectations
* `plots/{SPLIT}/kl` contains plots for the KL divergence
* `plots/{SPLIT}` contains all the scatter plots

## Citation

```
@misc{gad2024,
      title={Grammar-Aligned Decoding}, 
      author={Kanghee Park and Jiayu Wang and Taylor Berg-Kirkpatrick and Nadia Polikarpova and Loris D'Antoni},
      year={2024},
      eprint={2405.21047},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2405.21047}, 
}
```

## Acknowledgement

This project is built upon the [transformers-CFG](https://github.com/epfl-dlab/transformers-CFG) project.
