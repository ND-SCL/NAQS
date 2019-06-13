

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