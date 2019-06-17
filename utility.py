import logging
from config import ARCH_SPACE, QUAN_SPACE


def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(file_handler)
    return logger


def split_paras(paras):
    num_layers = len(paras)
    arch_paras = []
    quan_paras = []
    for i in range(num_layers):
        arch_paras.append({k: paras[i][k] for k in ARCH_SPACE})
        quan_paras.append({k: paras[i][k] for k in QUAN_SPACE})
    return arch_paras, quan_paras


class BestSamples(object):
    def __init__(self, length=5):
        self.length = length
        self.id_list = list(range(1, self.length+1))
        self.rollout_list = [[]] * self.length
        self.reward_list = [-1] * self.length

    def register(self, id, rollout, reward):
        for i in range(self.length):
            if reward > self.reward_list[i]:
                self.reward_list[i] = reward
                self.id_list[i] = id
                self.rollout_list[i] = rollout
                break

    def __repr__(self):
        return str(dict(zip(self.id_list, self.reward_list)))