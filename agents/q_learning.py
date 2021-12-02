import random

class QLearningMarioAgent:
    def __init__(self, actions, q_values=None, exploration_rate=0.8, learning_rate=0.99, discount=0.9999999, decay=0.99999):
        self.learning_rate = learning_rate
        self.discount = discount
        self.actions = actions
        self.exploration_rate = exploration_rate
        self.decay = decay
        if q_values:
            self.q_values = q_values
        else:
            self.q_values = {}

    def get_q_value(self, state, action):
        return self.q_values.get((state, action), 0.0)

    def get_q_values(self):
        return self.q_values

    def compute_value_from_q_value(self, state):
        return max(self.get_q_value(state, action) for action in range(self.actions))

    def compute_action_from_q_value(self, state):
        potential_actions = []
        for action in range(self.actions):
            if self.get_q_value(state, action) == self.compute_value_from_q_value(state):
                potential_actions.append(action)
        if not potential_actions:
            return None
        else:
            return random.choice(potential_actions)

    def get_action(self, state):
        if random.random() < self.exploration_rate:
            action = random.randint(0, self.actions - 1)
        else:
            action = self.compute_action_from_q_value(state)
        self.exploration_rate *= self.decay
        return action

    def update_q_values(self, state, action, next_state, reward):
        sample = reward + (self.discount * self.compute_value_from_q_value(next_state))
        self.q_values[(state, action)] = ((1 - self.learning_rate) * self.get_q_value(state, action)) \
                                         + (self.learning_rate * sample)