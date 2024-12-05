IS_PRINT = False
TARGET_TASK = "find_inv_bvsge_bvashr1_4bit"
DEBUG_LIST = [29898, 7922, 29899, 7692, 2437, 5135, 29879, 313, 21591, 25987, 29871, 29946, 876, 313, 29873, 313, 21591, 25987, 29871, 29946, 4961, 313, 21591, 25987, 29871, 29946, 29897, 313, 29890, 29894, 10052, 269, 876, 2]
PROB_TRUTH = []

def is_target_list(tensor):
    assert type(tensor) == list
    return tensor == DEBUG_LIST

def get_next_token(current):
    if len(current) < len(DEBUG_LIST) and current == DEBUG_LIST[:len(current)]:
        return DEBUG_LIST[len(current)]
    return None