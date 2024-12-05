import pickle
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
import sys
sys.path.append(".")
from transformers_gad.oracle.oracle_trie import Trie, TrieNode
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, FileType

# SPLIT = "binary"
# SPLIT = "SLIA"
SPLIT = "BV4"
# SPLIT = "CP"

RESULT_PATH = f"results/{SPLIT}"
PLOT_PATH = f"plots/{SPLIT}"

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def normalize(l):
    val_sum = sum(l)
    return [v / val_sum for v in l]

def KL(a, b):
    return scipy.stats.entropy(a, b)

def KL_dict(d):
    counts = [v[0] for v in d.values()]
    orig_probs = [v[1] for v in d.values()]

    counts = normalize(counts)
    orig_probs = normalize(orig_probs)

    return KL(counts, orig_probs)

def prob_explored(d):
    return sum([v[1] for v in d.values()])

def dict_sub(d1, d2):
    d = {}
    for k in d1.keys():
        if k in d2:
            v1 = d1[k]
            v2 = d2[k]
            d[k] = (v1[0] - v2[0], v1[1])
        else:
            d[k] = d1[k]
    return d

def check_equal(a, b):
    return abs(a - b) / max(a, b) < 1E-8

def parse_answer(answer):
    return tokens_to_str(answer['tokens']), answer['raw_likelihood']

def get_token_infos(answer_holder):
    token_dict = {}
    # size = None
    is_error = False
    for answers in answer_holder:
        # if size is None: size = len(answers)
        # assert size == len(answers)
        for answer in answers:
            res, prob = parse_answer(answer)
            if res not in token_dict:
                token_dict[res] = prob
            elif not check_equal(prob, token_dict[res]) and not is_error:
                print("prob mismatch ", res, prob, token_dict[res])
                is_error = True
    return token_dict

'''
def estimate_orig_dist(gad_answers, gcd_answers):
    tokens_count, prob_explored = count_appearance(gad_answers, dict(), 0)
    gad_tokens_count = dict(tokens_count)

    total_tokens_count, prob_explored = count_appearance(gcd_answers, tokens_count, prob_explored)
    
    gcd_tokens_count = dict_sub(total_tokens_count, gad_tokens_count)
    gad_tokens_count = dict_sub(total_tokens_count, gcd_tokens_count)


    assert(sum([v[0] for v in gcd_tokens_count.values()]) == sum([v[0] for v in gad_tokens_count.values()]))

    return gad_tokens_count, gcd_tokens_count
'''

def tokens_to_str(tokens):
    return "_".join(str(t) for t in tokens)

def count_appearance(answers, token_info):
    res = {token: (0, prob) for token, prob in token_info.items()}
    for answer in answers:
        tokens, prob = answer['tokens'], answer['raw_likelihood']
        token_str = tokens_to_str(tokens)
        v = res[token_str]
        res[token_str] = (v[0] + 1, prob)
    return res

def _get_ave_prob(answers, token_info):
    res = 0.0
    for answer in answers:
        token_str = tokens_to_str(answer["tokens"])
        res += token_info[token_str]
    return res / len(answers)

def count_appearance_all(answers, all_dict):
    tokens_count = {k:(0, v[1]) for (k, v) in all_dict.items()}
    tokens_history = []

    for answer in answers:
        tokens = answer['tokens']
        token_str = tokens_to_str(tokens)

        if token_str in tokens_count:
            v = tokens_count[token_str]
            tokens_count[token_str] = (v[0] + 1, v[1])
        else:
            print("Error")
            raise Exception("Unknown key")

        tokens_history.append(dict(tokens_count))

    kls = []
    interval = len(answers) // 4

    for i in range(len(answers) - interval):
        count_suffix = dict_sub(tokens_history[i + interval], tokens_history[i])
        kl = KL_dict(count_suffix)
        kls.append(kl)

    explored = prob_explored(all_dict)
    rel_probs = [answer['raw_likelihood'] / explored for answer in answers]

    return rel_probs, kls

def expectation(token_info):
    probs = list(token_info.values())
    probs = normalize(probs)
    return sum([v * v for v in probs])

def geo_mean(iterable):
    a = np.array(iterable)
    return a.prod()**(1.0/len(a))

def dist_to_line(x, y):
    p1 = np.array([0, 0])
    p2 = np.array([1, 1])
    p3 = np.array([x, y])
    return np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)

def read_answers(path):
    answers = []
    if not os.path.exists(path): return None

    with open(path, "r") as f:
        lines = f.readlines()
    
    for line in lines:
        if len(line) > 0:
            answers.append(json.loads(line))    

    return answers

def read_history(path):
    if not os.path.exists(path): return None
    with open(path, "r") as f:
        return json.load(f)

MCMC_ROUND = 100

def save_fig(prob_name):
    raw_mcmc = read_history(f"{RESULT_PATH}/{prob_name}/{prob_name}_raw_mcmc.jsonl")
    beam_mcmc = read_history(f"{RESULT_PATH}/{prob_name}/{prob_name}_beam_mcmc.jsonl")
    # gad = beam_mcmc[0] if beam_mcmc is not None else None
    gad = read_answers(f"{RESULT_PATH}/{prob_name}/{prob_name}_test_gad.jsonl")

    if any([x is None for x in [gad, raw_mcmc, beam_mcmc]]): return None

    token_info = get_token_infos([gad] + raw_mcmc + beam_mcmc)


    out_kl_path = f"{PLOT_PATH}/kl/{prob_name}.png"
    plt.cla()

    # draw gad kl
    interval_size = len(gad) // 4
    interval_info = count_appearance(gad[-interval_size:], token_info)
    gad_kl = KL_dict(interval_info)
    x_lim = max(len(raw_mcmc), len(beam_mcmc))
    plt.plot([1, x_lim], [gad_kl, gad_kl], '-g', label=f'AsAp[{len(gad) - interval_size}, {len(gad)}]')

    # draw raw_mcmc
    xs = list(range(1, 1 + len(raw_mcmc)))
    raw_ys = [KL_dict(count_appearance(answers, token_info)) for answers in raw_mcmc]
    plt.plot(xs, raw_ys, '--b', label='Raw MCMC')

    # draw projected_mcmc
    xs = list(range(1, 1 + len(beam_mcmc)))
    beam_ys = [KL_dict(count_appearance(answers, token_info)) for answers in beam_mcmc]
    plt.plot(xs, beam_ys, '--r', label='Projected MCMC')

    plt.legend()
    plt.savefig(out_kl_path)
    plt.close()

    ideal = expectation(token_info)
    explored_prob = sum(token_info.values())
    gad_val = _get_ave_prob(gad[-interval_size:], token_info) / explored_prob
    raw_val = _get_ave_prob(raw_mcmc[MCMC_ROUND], token_info) / explored_prob
    beam_val = _get_ave_prob(beam_mcmc[MCMC_ROUND], token_info) / explored_prob

    return ideal, gad_val, raw_val, beam_val
    
    # for x in xs:
    #     print(x, gad_kls[x], 'a')
    #     print(x, gcd_kls[x], 'b')
    
    # Probability 
    xs = range(len(gad_probs))
    out_prob_path = f"{PLOT_PATH}/prob/{prob_name}.png"
    plt.cla()

    # p1 = np.polyfit(xs, gad_probs, poly_order)
    sum_probs = 0
    gad_ys = []
    for i in xs:
        prob = gad_probs[i]
        sum_probs += prob
        gad_ys.append(sum_probs / (i + 1))

    plt.plot(xs, gad_probs, marker='o', linestyle='None', color='blue', label='ASAp', markersize=MARKER_SIZE)
    plt.plot(xs, gad_ys, '--b')

    sum_probs = 0
    beam_ys = []
    for i in xs:
        prob = beam_probs[i]
        sum_probs += prob
        beam_ys.append(sum_probs / (i + 1))

    # p2 = np.polyfit(xs, gcd_probs, poly_order)
    plt.plot(xs, beam_probs, marker='x', linestyle='None', color='red', label='Beam', markersize=MARKER_SIZE)
    plt.plot(xs, beam_ys, '--r')

    ideal = expectation_from_count(gad_tokens_count)
    plt.plot(xs, [ideal for _ in xs] , '--k', label='LLM')

    plt.legend()
    plt.savefig(out_prob_path)
    plt.close()

    return ideal, gad_ys[-1], beam_ys[-1]

def main():
    prob_names = [x[0] for x in os.walk(RESULT_PATH) if x[0] != RESULT_PATH]
    prob_names = [x.split("/")[-1] for x in prob_names]

    make_dir(PLOT_PATH)
    make_dir(PLOT_PATH + "/prob")
    make_dir(PLOT_PATH + "/kl")

    ideals, gad_lasts, raw_results, beam_results = [], [], [], []
    for prob_name in prob_names:
        info = save_fig(prob_name)

        if info is None: continue
        ideal, gad, raw, beam = info
        ideals.append(ideal)
        gad_lasts.append(gad)
        raw_results.append(raw)
        beam_results.append(beam)

    # Scatter
    plt.cla()

    plt.plot(ideals, gad_lasts, marker='o', linestyle='None', color='blue', label='ASAp[1500, 2000]')
    plt.plot(ideals, beam_results, marker='x', linestyle='None', color='red', label=f'Projected MCMC[{MCMC_ROUND}]')
    plt.plot(ideals, raw_results, marker='^', linestyle='None', color='orange', label=f'Raw MCMC[{MCMC_ROUND}]')

    print(gad_lasts)
    print(beam_results)
    print(raw_results)
    plt.axline((0, 0), slope=1)
    plt.xlim((0, 1))
    plt.ylim((0, 1))

    plt.legend()
    plt.savefig(PLOT_PATH + "/scatters.png")
    plt.close()

    # print("GCD: ", sum([dist_to_line(ideals[i], gcd_lasts[i]) for i in range(len(ideals))]))
    # print("GAD: ", sum([dist_to_line(ideals[i], gad_lasts[i]) for i in range(len(ideals))]))

if __name__ == "__main__":
    main()