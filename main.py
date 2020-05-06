import argparse
import csv
import logging
import os
import time

import torch

import child
import data
import backend
from controller import Agent
from config import ARCH_SPACE, QUAN_SPACE, CLOCK_FREQUENCY
from utility import BestSamples
from fpga.model import FPGAModel
import utility


# def get_args():
parser = argparse.ArgumentParser('Parser User Input Arguments')
parser.add_argument(
    'mode',
    default='nas',
    choices=['nas', 'joint', 'nested', 'quantization'],
    help="supported dataset including : 1. nas (default), 2. joint"
    )
parser.add_argument(
    '-d', '--dataset',
    default='CIFAR10',
    help="supported dataset including : 1. MNIST, 2. CIFAR10 (default)"
    )
parser.add_argument(
    '-l', '--layers',
    type=int,
    default=6,
    help="the number of child network layers, default is 6"
    )
parser.add_argument(
    '-rl', '--rLUT',
    type=int,
    default=1e5,
    help="the maximum number of LUTs allowed, default is 10000")
parser.add_argument(
    '-rt', '--rThroughput',
    type=float,
    default=1000,
    help="the minumum throughput to be achieved, default is 1000")
parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=30,
    help="the total epochs for model fitting, default is 30"
    )
parser.add_argument(
    '-ep', '--episodes',
    type=int,
    default=2000,
    help='''the number of episodes for training the policy network, default
        is 2000'''
    )
parser.add_argument(
    '-ep1', '--episodes1',
    type=int,
    default=1000,
    help='''the number of episodes for training the architecture, default
        is 1000'''
    )
parser.add_argument(
    '-ep2', '--episodes2',
    type=int,
    default=200,
    help='''the number of episodes for training the quantization, default
        is 500'''
    )
parser.add_argument(
    '-lr', '--learning_rate',
    type=float,
    default=0.2,
    help="learning rate for updating the controller, default is 0.2")
parser.add_argument(
    '-ns', '--no_stride',
    action='store_true',
    help="include stride in the architecture space, default is false"
    )
parser.add_argument(
    '-np', '--no_pooling',
    action='store_true',
    help="include max pooling in the architecture space, default is false"
    )
parser.add_argument(
    '-b', '--batch_size',
    type=int,
    default=128,
    help="the batch size used to train the child CNN, default is 128"
    )
parser.add_argument(
    '-s', '--seed',
    type=int,
    default=1,
    help="seed for randomness, default is 0"
    )
parser.add_argument(
    '-g', '--gpu',
    type=int,
    default=0,
    help="in single gpu mode the id of the gpu used, default is 0"
    )
parser.add_argument(
    '-k', '--skip',
    action='store_true',
    help="include skip connection in the architecture, default is false"
    )
parser.add_argument(
    '-a', '--augment',
    action='store_true',
    help="augment training data"
    )
parser.add_argument(
    '-m', '--multi-gpu',
    action='store_true',
    help="use all gpus available, default false"
    )
parser.add_argument(
    '-v', '--verbosity',
    type=int,
    choices=range(3),
    default=0,
    help="verbosity level: 0 (default), 1 and 2 with 2 being the most verbose"
    )
args = parser.parse_args()


if args.no_stride is True:
    if 'stride_height' in ARCH_SPACE:
        ARCH_SPACE.pop('stride_height')
    if 'stride_width' in ARCH_SPACE:
        ARCH_SPACE.pop('stride_width')

if args.no_pooling is True:
    if 'pool_size' in ARCH_SPACE:
        ARCH_SPACE.pop('pool_size')


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
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available()
                          else "cpu")
    print(f"using device {device}")
    dir = os.path.join(
        f'experiment',
        args.mode,
        'non_linear' if args.skip else 'linear',
        ('without' if args.no_stride else 'with') + ' stride, ' +
        ('without' if args.no_pooling else 'with') + ' pooling',
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
    logger.info(f"include stride: \t\t\t {not args.no_stride}")
    logger.info(f"include pooling: \t\t\t {not args.no_pooling}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"data augmentation: \t\t\t {args.augment}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    agent = Agent(ARCH_SPACE, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'), skip=args.skip)
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=args.augment)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy", "Time"]
                    )
    arch_id, total_time = 0, 0
    logger.info('=' * 50 + "Start exploring architecture space" + '=' * 50)
    logger.info('-' * len("Start exploring architecture space"))
    best_samples = BestSamples(5)
    for e in range(args.episodes):
        arch_id += 1
        start = time.time()
        arch_rollout, arch_paras = agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    arch_id, arch_rollout))
        model, optimizer = child.get_model(
            input_shape, arch_paras, num_classes, device,
            multi_gpu=args.multi_gpu, do_bn=True)
        _, arch_reward = backend.fit(
            model, optimizer, train_data, val_data,
            epochs=args.epochs, verbosity=args.verbosity)
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
        logger.info('-' * len("Start exploring architecture space"))
    logger.info(
        '=' * 50 + "Architecture sapce exploration finished" + '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()


def sync_search(device, dir='experiment'):
    dir = os.path.join(
        dir, f"rLut={args.rLUT}, rThroughput={args.rThroughput}")
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    filepath = os.path.join(dir, f"joint ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'joint'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"include stride: \t\t\t {not args.no_stride}")
    logger.info(f"include pooling: \t\t\t {not args.no_pooling}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"required # LUTs: \t\t\t {args.rLUT}")
    logger.info(f"required throughput: \t\t\t {args.rThroughput}")
    logger.info(f"Assumed frequency: \t\t\t {CLOCK_FREQUENCY}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"data augmentation: \t\t\t {args.augment}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    logger.info(f"quantization space: ")
    for name, value in QUAN_SPACE.items():
        logger.info(name + f": \t\t\t {value}")
    agent = Agent({**ARCH_SPACE, **QUAN_SPACE}, args.layers,
                  lr=args.learning_rate,
                  device=torch.device('cpu'), skip=args.skip)
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=args.augment)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy"] +
                    ["Partition (Tn, Tm)", "Partition (#LUTs)",
                    "Partition (#cycles)", "Total LUT", "Total Throughput"] +
                    ["Time"])
    child_id, total_time = 0, 0
    logger.info('=' * 50 +
                "Start exploring architecture & quantization space" + '=' * 50)
    best_samples = BestSamples(5)
    for e in range(args.episodes):
        logger.info('-' * 130)
        child_id += 1
        start = time.time()
        rollout, paras = agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled actions: {}".format(
                    child_id, rollout))
        arch_paras, quan_paras = utility.split_paras(paras)
        fpga_model = FPGAModel(rLUT=args.rLUT, rThroughput=args.rThroughput,
                               arch_paras=arch_paras, quan_paras=quan_paras)
        if fpga_model.validate():
            model, optimizer = child.get_model(
                input_shape, arch_paras, num_classes, device,
                multi_gpu=args.multi_gpu, do_bn=False)
            _, reward = backend.fit(
                model, optimizer, train_data, val_data, quan_paras=quan_paras,
                epochs=args.epochs, verbosity=args.verbosity)
        else:
            reward = 0
        agent.store_rollout(rollout, reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(child_id, rollout, reward)
        writer.writerow(
            [child_id] +
            [str(paras[i]) for i in range(args.layers)] +
            [reward] + list(fpga_model.get_info()) + [ep_time]
            )
        logger.info(f"Reward: {reward}, " +
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
    logger.info(
        '=' * 50 +
        "Architecture & quantization sapce exploration finished" +
        '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()


def nested_search(device, dir='experiment'):
    dir = os.path.join(
        dir, f"rLut={args.rLUT}, rThroughput={args.rThroughput}")
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    filepath = os.path.join(dir, f"nested ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'nested'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"include stride: \t\t\t {not args.no_stride}")
    logger.info(f"include pooling: \t\t\t {not args.no_pooling}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"required # LUTs: \t\t\t {args.rLUT}")
    logger.info(f"required throughput: \t\t\t {args.rThroughput}")
    logger.info(f"Assumed frequency: \t\t\t {CLOCK_FREQUENCY}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"data augmentation: \t\t\t {args.augment}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes1}")
    logger.info(f"quantization episodes: \t\t\t {args.episodes2}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"architecture space: ")
    for name, value in ARCH_SPACE.items():
        logger.info(name + f": \t\t\t\t {value}")
    logger.info(f"quantization space: ")
    for name, value in QUAN_SPACE.items():
        logger.info(name + f": \t\t\t {value}")
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=args.augment)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy"] +
                    ["Partition (Tn, Tm)", "Partition (#LUTs)",
                    "Partition (#cycles)", "Total LUT", "Total Throughput"] +
                    ["Time"])
    arch_agent = Agent(ARCH_SPACE, args.layers,
                       lr=args.learning_rate,
                       device=torch.device('cpu'), skip=args.skip)
    arch_id, total_time = 0, 0
    logger.info('=' * 50 +
                "Start exploring architecture space" + '=' * 50)
    best_arch = BestSamples(5)
    for e1 in range(args.episodes1):
        logger.info('-' * 130)
        arch_id += 1
        start = time.time()
        arch_rollout, arch_paras = arch_agent.rollout()
        logger.info("Sample Architecture ID: {}, Sampled arch: {}".format(arch_id, arch_rollout))
        model, optimizer = child.get_model(
            input_shape, arch_paras, num_classes, device,
            multi_gpu=args.multi_gpu, do_bn=False)
        backend.fit(
            model, optimizer, train_data, val_data,
            epochs=args.epochs, verbosity=args.verbosity)
        quan_id = 0
        best_quan_reward = -1
        logger.info('=' * 50 +
                "Start exploring quantization space" + '=' * 50)
        quan_agent = Agent(QUAN_SPACE, args.layers,
            lr=args.learning_rate,
            device=torch.device('cpu'), skip=False)
        for e2 in range(args.episodes2):
            quan_id += 1
            quan_rollout, quan_paras = quan_agent.rollout()
            fpga_model = FPGAModel(
                rLUT=args.rLUT, rThroughput=args.rThroughput,
                arch_paras=arch_paras, quan_paras=quan_paras)
            if fpga_model.validate():
                _, quan_reward = backend.fit(
                    model, optimizer,
                    val_data=val_data, quan_paras=quan_paras,
                    epochs=1, verbosity=args.verbosity)
            else:
                quan_reward = 0
            logger.info("Sample Quantization ID: {}, Sampled Quantization: {}, reward: {}".format(quan_id, quan_rollout, quan_reward))
            quan_agent.store_rollout(quan_rollout, quan_reward)
            if quan_reward > best_quan_reward:
                best_quan_reward = quan_reward
                best_quan_rollout, best_quan_paras = quan_rollout, quan_paras
        logger.info('=' * 50 +
                "Quantization space exploration finished" + '=' * 50)
        arch_reward = best_quan_reward
        arch_agent.store_rollout(arch_rollout, arch_reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_arch.register(arch_id, utility.combine_rollout(arch_rollout,best_quan_rollout, args.layers), arch_reward)
        writer.writerow(
            [arch_id] +
            [str(arch_paras[i]) + '\n' + str(best_quan_paras[i])
             for i in range(args.layers)] +
            [arch_reward] + list(fpga_model.get_info()) + [ep_time]
            )
        logger.info(f"Reward: {arch_reward}, " +
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e1+1)}")
        logger.info(f"Best Reward: {best_arch.reward_list[0]}, " +
                    f"ID: {best_arch.id_list[0]}, " +
                    f"Rollout: {best_arch.rollout_list[0]}")
    logger.info(
        '=' * 50 +
        "Architecture & quantization sapce exploration finished" +
        '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_arch}")
    csvfile.close()


def quantization_search(device, dir='experiment'):
    dir = os.path.join(
        dir, f"rLut={args.rLUT}, rThroughput={args.rThroughput}")
    if os.path.exists(dir) is False:
        os.makedirs(dir)
    filepath = os.path.join(dir, f"quantization ({args.episodes} episodes)")
    logger = get_logger(filepath)
    csvfile = open(filepath+'.csv', mode='w+', newline='')
    writer = csv.writer(csvfile)
    logger.info(f"INFORMATION")
    logger.info(f"mode: \t\t\t\t\t {'quantization'}")
    logger.info(f"dataset: \t\t\t\t {args.dataset}")
    logger.info(f"number of child network layers: \t {args.layers}")
    logger.info(f"include stride: \t\t\t {not args.no_stride}")
    logger.info(f"include pooling: \t\t\t {not args.no_pooling}")
    logger.info(f"skip connection: \t\t\t {args.skip}")
    logger.info(f"required # LUTs: \t\t\t {args.rLUT}")
    logger.info(f"required throughput: \t\t\t {args.rThroughput}")
    logger.info(f"Assumed frequency: \t\t\t {CLOCK_FREQUENCY}")
    logger.info(f"training epochs: \t\t\t {args.epochs}")
    logger.info(f"data augmentation: \t\t\t {args.augment}")
    logger.info(f"batch size: \t\t\t\t {args.batch_size}")
    logger.info(f"controller learning rate: \t\t {args.learning_rate}")
    logger.info(f"architecture episodes: \t\t\t {args.episodes}")
    logger.info(f"using multi gpus: \t\t\t {args.multi_gpu}")
    logger.info(f"architecture space: ")
    # for name, value in ARCH_SPACE.items():
    #     logger.info(name + f": \t\t\t\t {value}")
    logger.info(f"quantization space: ")
    for name, value in QUAN_SPACE.items():
        logger.info(name + f": \t\t\t {value}")
    agent = Agent(QUAN_SPACE, args.layers,
                lr=args.learning_rate, device=torch.device('cpu'), skip=False)
    train_data, val_data = data.get_data(
        args.dataset, device, shuffle=True,
        batch_size=args.batch_size, augment=args.augment)
    input_shape, num_classes = data.get_info(args.dataset)
    writer.writerow(["ID"] +
                    ["Layer {}".format(i) for i in range(args.layers)] +
                    ["Accuracy"] +
                    ["Partition (Tn, Tm)", "Partition (#LUTs)",
                    "Partition (#cycles)", "Total LUT", "Total Throughput"] +
                    ["Time"])
    child_id, total_time = 0, 0
    logger.info('=' * 50 +
                "Start exploring quantization space" + '=' * 50)
    best_samples = BestSamples(5)
    arch_paras = [
        {'filter_height': 3, 'filter_width': 3,
         'stride_height': 1, 'stride_width': 1,
         'num_filters': 64, 'pool_size': 1},
        {'filter_height': 7, 'filter_width': 5,
         'stride_height': 1, 'stride_width': 1,
         'num_filters': 48, 'pool_size': 1},
        {'filter_height': 5, 'filter_width': 5,
         'stride_height': 2, 'stride_width': 1,
         'num_filters': 48, 'pool_size': 1},
        {'filter_height': 3, 'filter_width': 5,
         'stride_height': 1, 'stride_width': 1,
         'num_filters': 64, 'pool_size': 1},
        {'filter_height': 5, 'filter_width': 7,
         'stride_height': 1, 'stride_width': 1,
         'num_filters': 36, 'pool_size': 1},
        {'filter_height': 3, 'filter_width': 1,
         'stride_height': 1, 'stride_width': 2,
         'num_filters': 64, 'pool_size': 2}]
    model, optimizer = child.get_model(
                input_shape, arch_paras, num_classes, device,
                multi_gpu=args.multi_gpu, do_bn=False)
    _, val_acc = backend.fit(
        model, optimizer, train_data = train_data, val_data=val_data,
        epochs=args.epochs, verbosity=args.verbosity)
    print(val_acc)
    for e in range(args.episodes):
        logger.info('-' * 130)
        child_id += 1
        start = time.time()
        quan_rollout, quan_paras = agent.rollout()
        logger.info("Sample Quantization ID: {}, Sampled actions: {}".format(
                    child_id, quan_rollout))
        fpga_model = FPGAModel(rLUT=args.rLUT, rThroughput=args.rThroughput,
                               arch_paras=arch_paras, quan_paras=quan_paras)
        if fpga_model.validate():
            _, reward = backend.fit(
                model, optimizer,
                val_data=val_data, quan_paras=quan_paras,
                epochs=1, verbosity=args.verbosity)
        else:
            reward = 0
        agent.store_rollout(quan_rollout, reward)
        end = time.time()
        ep_time = end - start
        total_time += ep_time
        best_samples.register(child_id, quan_rollout, reward)
        writer.writerow(
            [child_id] +
            [str(quan_paras[i]) for i in range(args.layers)] +
            [reward] + list(fpga_model.get_info()) + [ep_time]
            )
        logger.info(f"Reward: {reward}, " +
                    f"Elasped time: {ep_time}, " +
                    f"Average time: {total_time/(e+1)}")
        logger.info(f"Best Reward: {best_samples.reward_list[0]}, " +
                    f"ID: {best_samples.id_list[0]}, " +
                    f"Rollout: {best_samples.rollout_list[0]}")
    logger.info(
        '=' * 50 +
        "Quantization sapce exploration finished" +
        '=' * 50)
    logger.info(f"Total elasped time: {total_time}")
    logger.info(f"Best samples: {best_samples}")
    csvfile.close()


SCRIPT = {
    'nas': nas,
    'joint': sync_search,
    'nested': nested_search,
    'quantization': quantization_search
}

if __name__ == '__main__':
    import random
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    main()

