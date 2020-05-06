import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


input_size = 35
hidden_size = 35
num_layers = 2


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


class Agent():
    def __init__(self, para_space, para_num_layers, batch_size=5, lr=0.5,
                 device=torch.device('cpu')):
        self.para_space = para_space
        self.para_num_layers = para_num_layers
        self.num_paras_per_layer = len(self.para_space)
        self.para_names, self.para_values = zip(*self.para_space.items())
        self.seq_len = self.num_paras_per_layer * para_num_layers
        self.device = device
        self.batch_size = batch_size

        self.model = PolicyNetwork(tuple(len(v) for v in self.para_values),
                                   para_num_layers).to(device)
        self.optimizer = optim.SGD(self.model.parameters(), lr)
        # self.optimizer = optim.RMSprop(self.model.parameters(), 0.005)
        self.initial_h = torch.randn(num_layers, 1, hidden_size).to(device)
        self.initial_c = torch.randn(num_layers, 1, hidden_size).to(device)
        self.initial_input = torch.randint(
            len(self.para_values[-1]), (1, 1)
            ).to(device)
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
                    action = torch.multinomial(pi, 1)
                    x = action
                    rollout.append(action.item())
        return rollout, self._format_rollout(rollout)

    def forward(self):
        rollout_list = [torch.tensor(v).to(self.device)
                        for v in list(zip(*self.rollout_buffer))]
        x = [self.initial_input.repeat(1, self.batch_size)] + \
            [rollout_list[i].unsqueeze(0) for i in range(len(rollout_list)-1)]
        state = (
            self.initial_h.repeat(1, self.batch_size, 1),
            self.initial_c.repeat(1, self.batch_size, 1)
            )
        logits = []
        for i in range(self.para_num_layers):
            for j in range(self.num_paras_per_layer):
                y, state = self.model.sample(
                    x[i * self.num_paras_per_layer + j], state, i, j)
                logits.append(F.softmax(y, dim=-1))
        return logits

    def backward(self, logits):
        rollout_list = [torch.tensor(v).to(self.device)
                        for v in list(zip(*self.rollout_buffer))]
        reward_list = \
            torch.tensor(self.reward_buffer).unsqueeze(-1).to(self.device)
        E = torch.zeros(self.batch_size, 1).to(self.device)
        for i in range(self.para_num_layers):
            for j in range(self.num_paras_per_layer):
                logit = logits[i * self.num_paras_per_layer + j].squeeze(0)
                prob = torch.gather(logit, -1, rollout_list[
                    i * self.num_paras_per_layer + j].unsqueeze(-1))
                E += torch.log(prob)
        E = (E * reward_list).sum()
        if getattr(self, 'optimizer', None) is None:
            self.model.zero_grad()
            E.backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p += p.grad * lr
        else:
            loss = - E
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return - E

    def train_step(self, optimize=False):
        if len(self.rollout_buffer) > 0:
            logits = self.forward()
            loss = self.backward(logits)
        self.rollout_buffer.clear()
        self.reward_buffer.clear()
        self.initial_input = torch.randint_like(
            self.initial_input, len(self.para_values[-1]))
        self.initial_h = torch.randn_like(self.initial_h)
        self.initial_c = torch.randn_like(self.initial_c)
        return

    def store_rollout(self, rollout, reward):
        self.rollout_buffer.append(rollout)
        self.reward_buffer.append(reward-self.ema)
        self.reward_history.append(reward)
        if len(self.reward_history) > self.period:
            self.reward_history.pop(0)
        self._update_ema(reward)
        if len(self.rollout_buffer) == self.batch_size:
            self.train_step()

    def _update_ema(self, reward):
        if len(self.reward_history) < self.period:
            self.ema = np.mean(self.reward_history)
        else:
            multiplier = 2 / self.period
            self.ema = (reward - self.ema) * multiplier + self.ema

    def _format_rollout(self, actions):
        paras = []
        layer_paras = {}
        for i, v in enumerate(actions):
            para_index = i % self.num_paras_per_layer
            layer_paras[self.para_names[para_index]] = \
                self.para_values[para_index][v]
            if (i+1) % self.num_paras_per_layer == 0:
                paras.append(layer_paras)
                layer_paras = {}
        return paras

    def adjust_learning_rate(self, lr):
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr


if __name__ == '__main__':
    import torch
    import random
    seed = 0
    torch.manual_seed(seed)
    random.seed(seed)
    from config import ARCH_SPACE, QUAN_SPACE
    from controller_bench import controller_bench
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    controller_bench(
        {**ARCH_SPACE, **QUAN_SPACE}, 6, device, skip=False, epochs=200)
