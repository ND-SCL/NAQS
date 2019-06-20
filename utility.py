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
        para = paras[i]
        arch_para = {}
        quan_para = {}
        for name, _ in ARCH_SPACE.items():
            if name in para:
                arch_para[name] = para[name]
            if 'anchor_point' in para:
                arch_para['anchor_point'] = para['anchor_point']
        for name, _ in QUAN_SPACE.items():
            if name in para:
                quan_para[name] = para[name]
        if arch_para != {}:
            arch_paras.append(arch_para)
        if quan_para != {}:
            quan_paras.append(quan_para)
        if arch_paras == []:
            arch_paras = None
        if quan_paras == []:
            quan_paras = None
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



if __name__ == '__main__':
    paras = [
        {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 0
         'anchor_point': []},
        {'filter_height': 3, 'filter_width': 3, 'num_filters': 48,  # 1
         'anchor_point': [1]},
        {'filter_height': 3, 'filter_width': 3, 'num_filters': 36,  # 2
         'anchor_point': [1, 1]},
        {'filter_height': 5, 'filter_width': 5, 'num_filters': 36,  # 3
         'anchor_point': [1, 1, 1]},
        {'filter_height': 3, 'filter_width': 7, 'num_filters': 48,  # 4
         'anchor_point': [0, 0, 1, 1]},
        {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 5
         'anchor_point': [0, 1, 1, 1, 1]},
        {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 6
         'anchor_point': [0, 1, 1, 1, 1, 1]},
        {'filter_height': 7, 'filter_width': 3, 'num_filters': 36,  # 7
         'anchor_point': [1, 0, 0, 0, 0, 1, 1]},
        {'filter_height': 7, 'filter_width': 1, 'num_filters': 36,  # 8
         'anchor_point': [1, 0, 0, 0, 1, 1, 0, 1]},
        {'filter_height': 7, 'filter_width': 7, 'num_filters': 36,  # 9
         'anchor_point': [1, 0, 1, 1, 1, 1, 1, 1, 1]},
        {'filter_height': 5, 'filter_width': 7, 'num_filters': 36,  # 10
         'anchor_point': [1, 1, 0, 0, 1, 1, 1, 1, 1, 1]},
        {'filter_height': 7, 'filter_width': 7, 'num_filters': 48,  # 11
         'anchor_point': [1, 1, 1, 1, 0, 1, 0, 0, 0, 0, 1]},
        {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 12
         'anchor_point': [1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1]},
        {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 13
         'anchor_point': [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1]},
        {'filter_height': 7, 'filter_width': 5, 'num_filters': 48,  # 14
         'anchor_point': [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1]}]
    arch_paras, quan_paras = split_paras(paras)
    print(arch_paras)
    print()
    print(quan_paras)

