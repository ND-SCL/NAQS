import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


input_size = 35
hidden_size = 35
num_layers = 2
batch_size = 5
lr = 2
gamma = 1


class PolicyNetwork(nn.Module):
    def __init__(self, para_num_choices, para_num_layers):
        super(PolicyNetwork, self).__init__()
        self.para_num_choices = para_num_choices
        self.num_paras_per_layer = len(para_num_choices)
        self.para_num_layers = para_num_layers
        self.seq_len = self.num_paras_per_layer * self.para_num_layers
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers)
        for i in range(self.para_num_layers):
            for j in range(self.num_paras_per_layer):
                setattr(self, 'embedding_{}_{}'.format(i, j),
                        nn.Embedding(self.para_num_choices[
                            (j-1) % self.num_paras_per_layer], input_size)
                        )
                setattr(self, 'classifier{}_{}'.format(i, j),
                        nn.Linear(
                            hidden_size,
                            self.para_num_choices[j]
                            )
                        )

    def sample(self, x, state, layer_index=0, para_index=0):
        embedding = getattr(
            self, 'embedding_{}_{}'.format(layer_index, para_index))
        x = embedding(x)
        x, state = self.rnn(x, state)
        classifier = getattr(
            self, 'classifier{}_{}'.format(layer_index, para_index))
        x = classifier(x)
        # x = 4 * F.tanh(x)
        return x, state

    def forward(self, x, state):
        # the element shape of x is 1 x batch_size
        unscaled_logits = []
        for i in range(self.para_num_layers):
            for j in range(self.num_paras_per_layer):
                y, state = self.sample(
                    x[i*self.num_paras_per_layer + j], state, i, j)
                # the shape of y is 1 x batch_size x num_values
                unscaled_logits.append(y)
        return unscaled_logits


class Sigmoid(nn.Module):
    def __init__(self, hidden_size):
        super(Sigmoid, self).__init__()
        self.w_prev = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w_curr = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hj, hi):
        x = F.tanh(self.w_prev(hj) + self.w_curr(hi))
        return F.sigmoid(self.v(x))


class Agent():
    def __init__(self, para_space, para_num_layers,
                 device=torch.device('cpu')):
        self.para_space = para_space
        self.para_num_layers = para_num_layers
        self.num_paras_per_layer = len(self.para_space)
        self.para_names, self.para_values = zip(*self.para_space.items())
        self.seq_len = self.num_paras_per_layer * para_num_layers
        self.device = device

        self.model = PolicyNetwork(tuple(len(v) for v in self.para_values),
                                   para_num_layers).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr, momentum=0.0)
        self.initial_h = torch.randn(num_layers, 1, hidden_size).to(device)
        self.initial_c = torch.randn(num_layers, 1, hidden_size).to(device)
        self.initial_input = torch.randint(
            len(self.para_values[-1]), (1, 1)
            ).to(device)
        # input shape = seq_len x batch_size
        self.rollout_buffer = []
        self.reward_buffer = []
        self.reward_history = []
        self.period = 20
        self.ema = 0

    def rollout(self):
        x = self.initial_input
        state = (self.initial_h, self.initial_c)
        rollout = []
        with torch.no_grad():
            for i in range(self.para_num_layers):
                for j in range(self.num_paras_per_layer):
                    x, state = self.model.sample(x, state, i, j)
                    pi = F.softmax(torch.squeeze(x, dim=0), dim=-1)
                    print("pi shape ", pi.shape)
                    action = torch.multinomial(pi, 1)
                    x = action
                    rollout.append(action.item())
        return rollout  # self.model.rollout(x, state)

    def train_step(self, optimize=False):
        batch_size = len(self.rollout_buffer)
        rollout_list = [torch.tensor(v)
                        for v in list(zip(*self.rollout_buffer))]
        reward_batch = torch.tensor(self.reward_buffer)
        # reward_batch shape: (batch_size,)
        input_batch = [self.initial_input.expand(1, batch_size)] + \
            [rollout_list[i].unsqueeze(0) for i in range(len(rollout_list)-1)]
        # input_batch is a list of tensor of shape 1 x batch_size
        state = (self.initial_h.expand(num_layers, batch_size, hidden_size),
                 self.initial_c.expand(num_layers, batch_size, hidden_size))
        unscaled_logits = self.model.forward(input_batch, state)
        logit_list = [F.softmax(v.squeeze(), dim=-1) for v in unscaled_logits]
        # logit_list is a list of tensor of shape batch_size x num_paras
        loss = update_weights(self.model, logit_list,
                              rollout_list, reward_batch)
        self.rollout_buffer.clear()
        self.reward_buffer.clear()
        self.initial_h = torch.randn(
            num_layers, 1, hidden_size).to(self.device)
        self.initial_c = torch.randn(
            num_layers, 1, hidden_size).to(self.device)
        self.initial_input = torch.randint(
            len(self.para_values[-1]), (1, 1)).to(self.device)
        return loss

    def store_rollout(self, rollout, reward):
        self.rollout_buffer.append(rollout)
        self.reward_buffer.append(reward-self.ema)
        self.reward_history.append(reward)
        if len(self.reward_history) > self.period:
            self.reward_history.pop(0)
        self._update_ema(reward)

    def _update_ema(self, reward):
        if len(self.reward_history) < self.period:
            self.ema = np.mean(self.reward_history)
        else:
            multiplier = 2 / self.period
            self.ema = (reward - self.ema) * multiplier + self.ema


def update_weights(model, logit_list, rollout_list, reward_batch,
                   optimizer=None):
    loss_sum = torch.zeros(rollout_list[0].size(0), 1)
    layer_idx = 0
    for logit, rollout in zip(logit_list, rollout_list):
        prob = torch.gather(logit, -1, rollout.unsqueeze(1))
        log_prob = torch.log(prob)
        loss_sum += log_prob * gamma ** (len(logit_list)-1-layer_idx)
    reward_loss = loss_sum * reward_batch.unsqueeze(-1)
    loss = reward_loss.mean()
    if optimizer is None:
        model.zero_grad()
        loss.backward()
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p += p.grad * lr
    else:
        loss = - loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.item()


if __name__ == '__main__':
    import time

    torch.manual_seed(0)

    def get_reward(rollout):
        target = [4, 2, 0, 4, 1, 2, 5, 1, 2, 4, 2, 3, 2, 1, 5, 6, 3, 3, 5, 0,
                  2, 4, 4, 6]
        max_error = 0
        for i in range(len(rollout)):
            max_error += max((target[i] - 0), (6 - target[i]))
        error = 0
        max_error = 24
        for i in range(len(rollout)):
            error += abs(rollout[i] != target[i])
        return (max_error-error)/max_error

    from config import QUAN_SPACE

    def controller_bench():
        agent = Agent(QUAN_SPACE, 6)
        batch_size = 5
        max_epochs = 100
        best_rollout = 0
        best_reward = -100000
        start = time.time()
        for e in range(max_epochs):
            for i in range(batch_size):
                rollout = agent.rollout()
                reward = get_reward(rollout)
                if reward == 1:
                    print(e*batch_size + i)
                    # quit()
                if reward > best_reward:
                    best_reward = reward
                    best_rollout = rollout
                print("action: {}, reward: {}".format(rollout, reward))
                agent.store_rollout(rollout, reward)
            loss = agent.train_step()
            print("epoch {}, loss: {}".format(e, loss))
            print("best rollout {}, best reward: {}".format(
                best_rollout, best_reward))
        print("elasped time is {}".format(time.time()-start))

    # controller_bench()
    agent = Agent(QUAN_SPACE, 2)
    rollout = agent.rollout()
    print(rollout)
