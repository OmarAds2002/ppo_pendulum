# memory/rollout_buffer.py — remove xs entirely
class RolloutBuffer:
    def __init__(self):
        self.states    = []
        self.actions   = []
        self.rewards   = []
        self.dones     = []
        self.log_probs = []
        self.values    = []

    def store(self, state, action, reward, done, log_prob, value):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()