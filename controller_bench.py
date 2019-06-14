import random
import time
import torch

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


def get_reward(rollout, target):
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
    return (max_error-error)/max_error


def controller_bench(space, num_layers, device=torch.device('cpu'), skip=True,
                     epochs=200):
    batch_size = 5
    max_epochs = epochs
    best_rollout = []
    best_paras = []
    best_reward = -100000
    start = time.time()
    agent = ctrl.Agent(space, num_layers, batch_size, device, skip)
    target = get_target(space, num_layers, skip)
    for e in range(max_epochs):
        for i in range(batch_size):
            rollout, paras = agent.rollout()
            # print(rollout, paras)
            reward = get_reward(rollout, target)
            if reward == 1:
                print(e*batch_size + i)
                quit()
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


if __name__ == '__main__':
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
