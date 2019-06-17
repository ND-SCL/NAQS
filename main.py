import argparse
import csv
import logging
import os
import time

import torch

import child_pytorch
import child_keras
from controller import Agent
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
    default=6,
    help="the number of child network layers"
    )
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=30,
    help="the total epochs for model fitting"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=1000,
    help="the number of episodes for training the policy network"
    )
parser.add_argument(
    '-st', '--stride',
    action='store_true',
    help="include stride in the architecture space, default is false"
    )
# parser.add_argument(
#     '-lr', '--learning_rate',
#     type=float,
#     default=0.001,
#     help="the learning rate for training the CNN"
#     )
parser.add_argument(
    '-b', '--batch_size',
    type=int,
    default=5,
    help="the batch size used to train the controller"
    )
parser.add_argument(
    '-s', '--seed',
    type=int,
    default=1,
    help="seed for randomness"
    )
parser.add_argument(
    '-k', '--skip',
    action='store_true',
    help="include skip connection in the architecture, default is false"
    )
parser.add_argument(
    '-f', '--framework',
    choices=['keras', 'pytorch'],
    default='keras',
    help="framewor 'keras' or 'pytorch'")
# parser.add_argument(
#     '-a', '--augment',
#     action='store_true',
#     help="augment training data"
#     )
# parser.add_argument(
#     '-r', '--early_stop',
#     action='store_true',
#     help="the total epochs for model fitting"
#     )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    help="verbosity level: 0, 1 and 2 with 2 being the most verbose"
    )
args = parser.parse_args()


if args.stride is False:
    if 'stride_height' in ARCH_SPACE:
        ARCH_SPACE.pop('stride_height')
    if 'stride_width' in ARCH_SPACE:
        ARCH_SPACE.pop('stride_width')


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


framework = child_keras if args.framework == 'keras' else child_pytorch


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device {device}")
    dir = os.path.join(
        f'experiment ({args.framework})',
        'non_linear' if args.skip else 'linear',
        args.dataset + f"({args.layers} layers)"
        )
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    SCRIPT[args.mode](device, dir)


def nas(device, dir='experiment'):
    filepath = os.path.join(dir, f"nas ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'nas'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"verbosity: \t\t\t\t {args.verbosity}")
    logger.info(f"framework: \t\t\t\t {args.framework}")
    logger.info(f"include stride: \t\t\t\t {args.stride}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    agent = Agent(
        ARCH_SPACE, args.layers, args.batch_size,
        device=torch.device('cpu'),
        skip=args.skip)
    child = framework.ChildCNN(dataset=args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy", "Time"]
                    )
    arch_id = 0
    total_time = 0
    logger.info('=' * 60 + "Start exploring architecture space" + '=' * 60)
    logger.info('-' * 180)
    best_samples = BestSamples(5)
    for e in range(args.episodes):
        arch_id += 1
        start = time.time()
        arch_rollout, arch_paras = agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    arch_id, arch_rollout))
        child.update_architecture(arch_paras)
        _, arch_reward = child.fit(
            epochs=args.epochs,
            validate=True,
            quantize=False,
            verbosity=args.verbosity)
        child.collect_garbage()
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
        logger.info('-' * 180)
    logger.info(
        '=' * 60 + "Architecture sapce exploration finished" + '=' * 60)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()


SCRIPT = {
    'nas': nas
}

if __name__ == '__main__':
    import random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main()
