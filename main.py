import argparse
import csv
import logging
import os
import time

import torch

import backend
import child
import data
import controller_nl as ctrl
from config import ARCH_SPACE, QUAN_SPACE
from utility import BestSamples


# def get_args():
parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    '-m', '--mode',
    default='nas',
    choices=['nas', 'joint'],
    help="supported dataset including : 1. nas, 2. joint"
    )
parser.add_argument(
    '-d', '--dataset',
    default='MNIST',
    help="supported dataset including : 1. MNIST, 2. CIFAR10"
    )
parser.add_argument(
    '-l', '--layers',
    type=int,
    default=1,
    help="the number of child network layers"
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=40,
    help="the total epochs for model fitting"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=1000,
    help="the number of episodes for training the policy network"
    )
parser.add_argument(
    '-s', '--early_stop',
    action='store_true',
    help="the total epochs for model fitting"
    )
parser.add_argument(
    '-v', '--verbose',
    action='store_false',
    help="Verbose or succint"
    )
args = parser.parse_args()


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    dir = os.path.join(
        'experiment',
        args.dataset + f"({args.layers} layers)"
        )
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    SCRIPT[args.mode](device, dir)


batch_size = 5


def nas(device, dir='experiment'):
    train_data, val_data = data.get_data(args.dataset, device)
    input_shape, num_classes = data.get_info(args.dataset)
    agent = ctrl.get_agent(ARCH_SPACE, args.layers, batch_size, device)
    filepath = os.path.join(dir, f"nas ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'nas'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"architecture spacce: \n")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    logger.info(f"architecture episodes: \t\t\t\t {args.episodes}")
    logger.info(f"early stop: \t\t\t {args.early_stop}")
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy", "Time"]
                    )
    arch_id = 0
    total_time = 0
    logger.info('=' * 20 + "Start exploring architecture space" + '=' * 20)
    best_samples = BestSamples(5)
    for e in range(args.episodes):
        arch_id += 1
        start = time.time()
        arch_rollout, arch_paras = agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    arch_id, arch_rollout))
        model, optimizer = child.get_model(
            input_shape, arch_paras, num_classes, device
            )
        arch_reward = backend.fit(
            model, optimizer,
            train_data, val_data,
            epochs=args.epochs,
            verbose=args.verbose,
            early_stop=args.early_stop
            )
        agent.store_rollout(arch_rollout, arch_reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(arch_id, arch_rollout, arch_reward)
        writer.writerow([arch_id] +
                        [str(arch_paras[i]) for i in range(args.layers)] +
                        [arch_reward] +
                        [ep_time])
        logger.info(f"Architecture Reward: {arch_reward}, " +
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
    logger.info(
        '=' * 20 + "Architecture sapce exploration finished" + '=' * 20)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()


SCRIPT = {
    'nas': nas
}

if __name__ == '__main__':
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    main()
