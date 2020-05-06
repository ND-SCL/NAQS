import matplotlib.pyplot as plt
import random
import time
import torch
from fpga.model import FPGAModel
import utility


import controller as ctrl


def get_target(para_space, num_layers, skip=True):
    _, para_values = zip(*para_space.items())
    num_choices = [len(v) for v in para_values]
    target = []
    for i in range(num_layers):
        if skip:
            target_anchor = []
            for j in range(i):
                target_anchor.append(random.randint(0, 1))
            target.append(target_anchor)
        for v in num_choices:
            target.append(random.randint(0, v-1))
    return target


def get_reward(rollout, quan_paras, target):
    error = 0
    for r, t in zip(rollout, target):
        if type(r) is list:
            for rr, tt in zip(r, t):
                error += abs(rr != tt) + random.normalvariate(0, 0.0)
        else:
            error += abs(r != t)
    max_error = 0
    for t in target:
        if type(t) is list:
            max_error += len(t)
        else:
            max_error += 1
    # return (max_error-error)/max_error * 0.50 + 0.40
    if bad_quan(quan_paras, target):
        return 0.1 + random.normalvariate(0, 0.1)
    else:
        return (max_error-error)/max_error * 0.8 + 0.1
    # return (max_error-error)/max_error


def bad_quan(quan_paras, target):
    q = []
    for l in quan_paras:
        for k, v in l.items():
            q.append(v)
    return (sum(q)/len(q) < 2) and (random.randint(0, 1)>0)



def plot(reward_history):
    plt.plot(range(len(reward_history)), reward_history)
    plt.show()


def controller_bench(space, num_layers, device=torch.device('cpu'), skip=True,
                     epochs=200):
    lr = 0.2
    batch_size = 5
    max_epochs = epochs
    best_rollout = []
    best_paras = []
    best_reward = -100000
    start = time.time()
    agent = ctrl.Agent(space, num_layers, batch_size,
        lr=lr, device=device, skip=skip)
    target = get_target(space, num_layers, skip)
    reward_history = []
    for e in range(max_epochs):
        # if e == 100:
        #     agent.lr_decay(0.3)
        for i in range(batch_size):
            rollout, paras = agent.rollout()
            print(agent.agent.ema)
            # print(rollout, paras)
            arch_paras, quan_paras = utility.split_paras(paras)
            # fpga_model = FPGAModel(
            #     rLUT=100000, rThroughput=1000,
            #     arch_paras=arch_paras, quan_paras=quan_paras)
            reward = get_reward(rollout, quan_paras, target)
            reward_history.append(reward)
            # if reward == 1:
            #     print(e*batch_size + i)
            #     quit()
            if reward > best_reward:
                best_reward = reward
                best_rollout = rollout
                best_paras = paras
                # print(best_rollout, best_paras)
            print("action: {}, reward: {}".format(rollout, reward))
            agent.store_rollout(rollout, reward)
        # E = agent.train_step()
        print("epoch {}".format(e))
        print(f"best rollout {best_rollout}, " +
              f"best architecture: {best_paras}, " +
              f"best reward: {best_reward}")
    print("elasped time is {}".format(time.time()-start))
    print("target: {}".format(target))
    plot(reward_history)


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
