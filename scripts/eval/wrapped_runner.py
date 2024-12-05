import os
import sys

command = "CUDA_VISIBLE_DEVICES=0 python3 scripts/eval/eval_gad.py"

for path in os.listdir("results/BV4"):
    if "find_inv" not in path: continue
    os.system(command + " " + path)
