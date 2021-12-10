import random

from agents.q_learning_agent import QLearningMarioAgent


class ApproxQLearningMarioAgent(QLearningMarioAgent):

    def __init__(self, actions, weights=None, exploration_rate=0.8, learning_rate=0.6, discount=0.9, decay=0.99):
        super().__init__(actions, exploration_rate=exploration_rate, learning_rate=learning_rate, discount=discount, decay=decay)
        if weights:
            self.weights = weights
        else:
            # if not given, initialize empty dictionary
            self.weights = {"coins": 0.0,
                            "flag_get": 0.0,
                            "life": 0.0,
                            "score": 0.0,
                            "stage": 0.0,
                            "status": 0.0,
                            "time": 0.0,
                            "world": 0.0,
                            "x_pos": 0.0,
                            "y_pos": 0.0,
                            }

    def get_weights(self):
        return self.weights

    def get_q_value(self, state, action):
        """

        """
        q_value = 0
        for key, value in state:
            q_value += self.weights[key] * value

        return q_value

    def update(self, state, action, next_state, reward):
        difference = reward + self.discount * self.compute_value_from_q_value(next_state) - self.get_q_value(state, action)

        # features_vect = self.featExtractor.getFeatures(state, action)
        for feature, value in state.items():
            self.weights[feature] += self.learning_rate * difference * value




    def get_q_values(self):
        """
        Returns the map of State, Action pairs to Q-values
        """
        return self.q_values

    def compute_value_from_q_value(self, state):
        """
        Computes the value of the given state.

        :param state: The state to find value for.
        :return (float): The highest q-value over all actions possible from this state
        """
        return max(self.get_q_value(state, action) for action in range(self.actions))

    def compute_action_from_q_value(self, state):
        """
        Computes the best action for a given state using the currently stored q-values.

        :param state: The state to find an action for
        :return: the action with the highest q-value for this state
        """
        potential_actions = []
        for action in range(self.actions):
            if self.get_q_value(state, action) == self.compute_value_from_q_value(state):
                potential_actions.append(action)
        if not potential_actions:
            return None
        else:
            return random.choice(potential_actions)

    def get_action(self, state):
        """
        Gets the action to perform next in the environment using exploration rate as the probability of random action
        chosen over best action.

        :param state: The state to determine an action for.
        :return: the action to take from the current state.
        """
        if random.random() < self.exploration_rate:
            action = random.randint(0, self.actions - 1)
        else:
            action = self.compute_action_from_q_value(state)
        self.exploration_rate *= self.decay
        return action
