import copy
import math
from os import device_encoding

import torch.nn.functional as F
import tqdm

import torch
import logging
from transformers.generation.logits_process import (
    LogitsProcessor,
    LOGITS_PROCESSOR_INPUTS_DOCSTRING,
)
from transformers.utils import add_start_docstrings

import debug
from transformers_gad.grammar_utils import IncrementalGrammarConstraint
from transformers_gad.recognizer import AcceptState
import random

RAW, MASKED = "raw", "masked"
node_index = 0

class SampleNode:
    def __init__(self, prefix, raw_prob, masked_prob, parent, last_token, state):
        # print("newnode ", prefix, raw_prob)
        global node_index
        self.index = node_index
        node_index += 1
        self.prefix = prefix
        self.prefix_prob = {RAW: float(raw_prob), MASKED: float(masked_prob)}
        self.sum_prob = None
        self.parent = parent
        self.children = {}
        self.step_prob = {}
        self.last_token = last_token
        self.top_leaves = None
        self.state = state

    def is_new(self):
        return self.sum_prob is None

    def to_string(self, tokenizer, skip_size=0):
        return tokenizer.decode(self.prefix[skip_size:])

    def to_string_all(self, tokenizer):
        return f"{tokenizer.decode(self.prefix)}@{self.prefix_prob}"

    def sample_next(self, forbid=None):
        items = [token for token in self.children if token != forbid]
        weights = [self.step_prob[token] for token in items]
        #print(items, weights, "forbid", forbid)
        token = random.choices(items, weights=weights, k=1)[0]
        return token

    def get_child(self, token, is_strict=True):
        assert self.children is not None
        if is_strict: assert token in self.children
        return self.children[token] if token in self.children else None

    def move_down(self, inp_ids, eos, is_strict=True):
        result = self
        while result.prefix.shape[0] < inp_ids.shape[0] and result.last_token != eos:
            current_token = int(inp_ids[result.prefix.shape[0]])
            result = result.get_child(current_token, is_strict)
            if result is None: return result
        return result

    def insert_child(self, token, step_prob, constraint: IncrementalGrammarConstraint):
        token_tensor = torch.tensor([token], device=self.prefix.device)
        new_prefix = torch.cat((self.prefix, token_tensor), dim=0)
        next_state = constraint._consume_token_id(token, self.state)
        child = SampleNode(new_prefix, self.prefix_prob[RAW] * step_prob,
                           self.prefix_prob[MASKED] * step_prob / self.sum_prob, self, token, next_state)
        self.children[token] = child
        self.step_prob[token] = step_prob

    def extend(self, acc, raw_scores, constraint, debug_token=None):
        self.sum_prob = 0.0
        raw_prob = F.softmax(raw_scores, dim=-1)
        tokens = [x[0] for x in acc.nonzero().cpu().tolist()]
        for token in tokens:
            prob = float(raw_prob[token])
            self.sum_prob += prob
        assert self.sum_prob > 0

        for token in tokens:
            prob = float(raw_prob[token])
            self.insert_child(token, prob, constraint)
            # if token == debug_token:
            #    debug.PROB_TRUTH.append(float(prob))
            #     print("    next prob", int(token), float(prob))


class MCMCLogitsProcessor(LogitsProcessor):
    def __init__(self, grammar_constraint, parse_start_index, tokenizer, root):
        # Parser variables
        self.grammar_constraint = grammar_constraint
        self.parse_start_index = parse_start_index
        self.root = root
        self.tokenizer = tokenizer

        # To start with a longer prefix in enumerative search
        self.generate_start_index = None
        self.generated_tokens = None
        self.prefix_restriction = []

    def reset(self):
        self.reset_parser()
        self.prefix_restriction = None

    def reset_parser(self):
        if self.grammar_constraint.is_incremental:
            self.grammar_constraint.reset()

        self.generate_start_index = None
        self.generated_tokens = None

    def mask_scores(self, scores, device, nodes, expected=None):
        """
        resolve each stack to a tensor of True/False for each token
        indicating acceptance
        """

        masked_scores = torch.full(scores.shape, -math.inf).to(device)

        for i, node in enumerate(nodes):
            if node is None: continue
            if node.is_new():
                acceptance = self.grammar_constraint.filter_vocab(node.state, device)

                current = node.prefix[self.root.prefix.shape[0]:].cpu().tolist()
                node.extend(acceptance, scores[i], self.grammar_constraint)
            if expected is not None:
                masked_scores[i, expected] = scores[i, expected]
            else:
                for token in node.children:
                    masked_scores[i, token] = scores[i, token]
        # print(masked_scores[:, :6])
        return masked_scores

    def process_scores(self, input_ids, scores):
        # we dynamically create stacks at the first call, so that we know the batch size and beam size
        #print("current process", input_ids.shape)
        #for i in range(input_ids.shape[0]):
        #    print("  ", input_ids[i, self.root.prefix.shape[0]:].cpu().tolist())
        #    print("  ", input_ids[i].shape[0], self.tokenizer.decode(input_ids[i, self.root.prefix.shape[0]:]))
        print("current tokens", input_ids[0, self.root.prefix.shape[0]:].cpu().tolist())

        nodes = [
            self.root.move_down(inp, self.tokenizer.eos_token_id, False) for inp in input_ids
        ] if self.root is not None else None

        # assume the generation starts from the same index
        if self.generate_start_index is None:
            # the default is the end of input sequence of tokens
            self.generate_start_index = self.parse_start_index \
                if self.parse_start_index else input_ids.size(1)
        self.generated_tokens = input_ids[:, self.generate_start_index:]

        assert input_ids.shape[1] >= self.parse_start_index

        if input_ids.shape[1] < len(self.prefix_restriction):
            expected = self.prefix_restriction[input_ids.shape[1]]
        else:
            expected = None

        masked_scores = self.mask_scores(scores, scores.device, nodes, expected)


        return masked_scores

    @add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(
            self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        # print(input_ids)
        # ONE, ZERO = 29896, 29900
        # print(f"{input_ids.cpu().tolist()[0][25:]}: {float(scores[0, ZERO].cpu())}, {float(scores[0, ONE].cpu())}")
        return self.process_scores(input_ids, scores)


    def set_prefix_restriction(self, prefix):
        # print("reset")
        self.reset()
        self.prefix_restriction = prefix

    def get_accepted_tokens(self, acceptance):
        """
        Get the indices of accepted tokens and their corresponding string values for each item in the batch.

        Parameters:
        - acceptance (torch.Tensor): A boolean tensor indicating accepted tokens for each item in the batch.
        """
        batch_size, _ = acceptance.shape
        acceptance_np = acceptance.cpu().numpy()
        accepted_x, accepted_y = acceptance_np.nonzero()

        # Initialize the dictionary with empty lists for indices
        accepted_token_indices = {i: [] for i in range(batch_size)}
        for x, y in zip(accepted_x, accepted_y):
            accepted_token_indices[x].append(y)

        # Convert token IDs to tokens
        accepted_tokens = {
            i: [self.grammar_constraint.tokenizer.decode([token_id]) for token_id in token_ids]
            for i, token_ids in accepted_token_indices.items()
        }

        return accepted_tokens

def _get_meaningful_parents(node):
    parent_list = []
    while node.parent is not None:
        last_token = node.last_token
        node = node.parent
        if len(node.children) > 1:
            parent_list.append((node, last_token))
    return parent_list

def _get_lca(x: SampleNode, y: SampleNode):
    upper_x, upper_y = x, y
    while upper_x.parent != upper_y.parent:
        if upper_x.prefix.shape[0] < upper_y.prefix.shape[0]:
            upper_y = upper_y.parent
        else:
            upper_x = upper_x.parent
    assert upper_x != upper_y
    return upper_x, upper_y

class RawMCMCHolder:
    def __init__(self, inp_ids, generator, tokenizer, constraint, device):
        self.generator = generator
        self.tokenizer = tokenizer
        self.device = device
        self.history = []

        self.root = SampleNode(inp_ids, 1.0, 1.0, None, None, constraint.string_recognizer.get_initial_accept_state())
        self.processor = MCMCLogitsProcessor(constraint, inp_ids.shape[0], self.tokenizer, self.root)

    def draw(self, node: SampleNode = None):
        if node is None: node = self.root
        while not node.is_new():
            token = node.sample_next()
            node = node.children[token]
            if token == self.tokenizer.eos_token_id:
                return node
        self.processor.set_prefix_restriction(node.prefix)
        res = self.generator(self.processor)
        return self.root.move_down(res, self.tokenizer.eos_token_id)

    def _get_transition_prob(self, x: SampleNode, y: SampleNode):
        parents_x = _get_meaningful_parents(x)
        parents_y = _get_meaningful_parents(y)
        assert len(parents_x) > 0 and len(parents_y) > 0

        upper_x, upper_y = _get_lca(x, y)
        token_x, token_y = upper_x.last_token, upper_y.last_token
        lca = upper_x.parent

        trans_xy = 1 / len(parents_x) * y.prefix_prob[MASKED] / ((lca.sum_prob - lca.step_prob[token_x]) / lca.sum_prob) / lca.prefix_prob[MASKED]
        trans_yx = 1 / len(parents_y) * x.prefix_prob[MASKED] / ((lca.sum_prob - lca.step_prob[token_y]) / lca.sum_prob) / lca.prefix_prob[MASKED]
        # print()
        # print("trans between ", x.to_string(self.tokenizer, self.root.prefix.shape[0]), y.to_string(self.tokenizer, self.root.prefix.shape[0]))
        # print("  ", lca.to_string(self.tokenizer, self.root.prefix.shape[0]))
        # print("  ", x.prefix_prob[MASKED], y.prefix_prob[MASKED], lca.prefix_prob[MASKED])
        # print("  ", trans_xy, trans_yx)
        return trans_xy, trans_yx

    def _get_accept_prob(self, x: SampleNode, y: SampleNode):
        trans_xy, trans_yx = self._get_transition_prob(x, y)
        assert max(trans_xy, trans_yx) < 1 + 1e-10 and min(trans_xy, trans_yx) > -1e-10

        accept_prob = y.prefix_prob[RAW] * trans_yx / x.prefix_prob[RAW] / trans_xy
        return min(accept_prob, 1.0)

    def mutate(self, node_x):
        parents = _get_meaningful_parents(node_x)
        if len(parents) == 0: return None
        parent, token_x = random.choice(parents)
        token_y = parent.sample_next(token_x)
        # print("start draw")
        node_y = self.draw(parent.children[token_y])
        # print("finish draw")
        return node_y

    def _is_accept(self, node_x, node_y):
        acc_xy = self._get_accept_prob(node_x, node_y)

        # DEBUG
        trans_xy, trans_yx = self._get_transition_prob(node_x, node_y)
        acc_yx = self._get_accept_prob(node_y, node_x)
        #print("weight from x to y", node_x.prefix_prob[RAW] * trans_xy * acc_xy, acc_xy)
        #print("weight from y to x", node_y.prefix_prob[RAW] * trans_yx * acc_yx, acc_yx)
        # print("from", node_x.to_string(self.tokenizer, self.root.prefix.shape[0]), node_x.prefix_prob[RAW])
        # print("  to", node_y.to_string(self.tokenizer, self.root.prefix.shape[0]), node_y.prefix_prob[RAW])
        #print(trans_xy, trans_yx)
        # print()
        assert abs(node_x.prefix_prob[RAW] * trans_xy * acc_xy - node_y.prefix_prob[RAW] * trans_yx * acc_yx) < 1e-15

        # self.history.append(acc_xy)
        # print("  acc", acc_xy)
        if random.random() < acc_xy:
            return True
        else:
            return False

    def mcmc(self, iter_num):
        x = self.draw()
        print("init", self.tokenizer.decode(x.prefix[self.root.prefix.shape[0]:]), x.prefix_prob[RAW], x.prefix_prob[MASKED])
        while len(self.history) < iter_num: self.history.append([])
        for index in tqdm.tqdm(range(iter_num), leave=False):
            # print("current node", x.to_string(self.tokenizer, self.root.prefix.shape[0]), x.prefix_prob[RAW])
            y = self.mutate(x)
            if y is not None and self._is_accept(x, y):
                x = y
                print("accepted to", self.tokenizer.decode(x.prefix[self.root.prefix.shape[0]:]), x.prefix_prob[RAW], x.prefix_prob[MASKED])
            info = {"tokens": x.prefix[self.root.prefix.shape[0]:].cpu().tolist(), "raw_likelihood": x.prefix_prob["raw"]}
            self.history[index].append(info)
        return x


class BeamMCMC(RawMCMCHolder):
    def __init__(self, inp_ids, generator, searcher, tokenizer, constraint, device, beam_weight=0.5, beam_num=10):
        super().__init__(inp_ids, generator, tokenizer, constraint, device)
        self.searcher = searcher
        self.beam_weight = beam_weight
        self.beam_num = beam_num

        self.beam_root = SampleNode(inp_ids, 1.0, 1.0, None, None, constraint.string_recognizer.get_initial_accept_state())
        self.beam_processor = MCMCLogitsProcessor(constraint, inp_ids.shape[0], self.tokenizer, self.beam_root)
        self._get_top_leaves(self.root)


    def _create_leaf_prob(self, beam_node: SampleNode):
        leaf = self.root.move_down(beam_node.prefix, self.tokenizer.eos_token_id, False)
        if leaf is not None: return leaf
        self.processor.set_prefix_restriction(beam_node.prefix)
        res = self.generator(self.processor)
        return self.root.move_down(res, self.tokenizer.eos_token_id)

    def _get_top_leaves(self, node: SampleNode):
        if node.top_leaves is not None: return
        # print("beam size", self.beam_num)
        # print("sample for", self.tokenizer.decode(node.prefix[self.root.prefix.shape[0]:]))
        self.beam_processor.set_prefix_restriction(node.prefix)
        # print("start search")
        res = self.searcher(self.beam_processor, self.beam_num)
        # print("end search", res.shape)

        node.top_leaves = {}
        for sample in res:
            # print("  sample", sample[self.root.prefix.shape[0]:].cpu().tolist())
            beam_leaf = self.beam_root.move_down(sample, self.tokenizer.eos_token_id)
            leaf = self._create_leaf_prob(beam_leaf)
            if leaf.index not in node.top_leaves:
                node.top_leaves[leaf.index] = leaf
                # print("  top", self.tokenizer.decode(leaf.prefix[self.root.prefix.shape[0]:]), leaf.prefix_prob[RAW])

    def mutate(self, node_x):
        if random.random() > self.beam_weight:
            return super().mutate(node_x)

        parents = _get_meaningful_parents(node_x)
        if len(parents) == 0: return None
        parent, token_x = random.choice(parents)
        token_y = parent.sample_next(token_x)
        branch: SampleNode = parent.children[token_y]
        self._get_top_leaves(branch)

        choices, weights = [], []
        for node in branch.top_leaves.values():
            choices.append(node)
            weights.append(node.prefix_prob[RAW])

        return random.choices(choices, weights=weights, k=1)[0]

    def _get_transition_prob(self, x: SampleNode, y: SampleNode):
        raw_xy, raw_yx = super()._get_transition_prob(x, y)

        upper_x, upper_y = _get_lca(x, y)
        lca = upper_x.parent
        token_x, token_y = upper_x.last_token, upper_y.last_token
        parents_x = _get_meaningful_parents(x)
        parents_y = _get_meaningful_parents(y)

        self._get_top_leaves(upper_x)
        self._get_top_leaves(upper_y)
        if y.index not in upper_y.top_leaves:
            trans_xy = 0.0
        else:
            total = 0.0
            for node in upper_y.top_leaves.values(): total += node.prefix_prob[RAW]
            trans_xy = 1.0 / len(parents_x) * lca.step_prob[token_y] / (lca.sum_prob - lca.step_prob[token_x]) * y.prefix_prob[RAW] / total
        if x.index not in upper_x.top_leaves:
            trans_yx = 0.0
        else:
            total = 0.0
            for node in upper_x.top_leaves.values(): total += node.prefix_prob[RAW]
            trans_yx = 1.0 / len(parents_y) * lca.step_prob[token_x] / (lca.sum_prob - lca.step_prob[token_y]) * x.prefix_prob[RAW] / total

        trans_xy = trans_xy * self.beam_weight + raw_xy * (1 - self.beam_weight)
        trans_yx = trans_yx * self.beam_weight + raw_yx * (1 - self.beam_weight)
        return trans_xy, trans_yx


