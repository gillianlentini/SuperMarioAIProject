import random


class QLearningMarioAgent:
    """
    Class representing a traditional Q-Learning Super Mario Agent.
    """

    def __init__(self, actions, q_values=None, exploration_rate=0.8, learning_rate=0.6, discount=0.9, decay=0.99999):
        """
        Initializes the Q-Learning agent.

        :param actions (list): representing possible actions.
        :param q_values (dict): represents the current q-values with state and action mapped to q-value
        :param exploration_rate (float): representing the rate of exploration (taking random action)
        :param learning_rate (float): learning rate
        :param discount (float): discount factor
        :param decay (float): represents the rate to decay frequency of random action by
        """

        self.learning_rate = learning_rate
        self.discount = discount
        self.actions = actions
        self.exploration_rate = exploration_rate
        self.decay = decay
        if q_values:
            self.q_values = q_values
        else:
            # if not given, initialize empty dictionary
            self.q_values = {}

    def get_q_value(self, state, action):
        """
        Gets the Q-value for a given state action pair.

        :param state: Represents the state to get a q-value for
        :param action: Represents the action to get the q-value for
        :return: The stored q-value or 0.0 if it does not exist
        """
        return self.q_values.get((state, action), 0.0)

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

    def update(self, state, action, next_state, reward):
        """
        Updates the q-value of the given state, action pair based on the
        observation of the next state reached and reward.

        :param state: the state you were in
        :param action: the action that was performed
        :param next_state: the state that was reached
        :param reward: the reward that was gained
        """
        sample = reward + (self.discount * self.compute_value_from_q_value(next_state))
        self.q_values[(state, action)] = ((1 - self.learning_rate) * self.get_q_value(state, action)) \
                                         + (self.learning_rate * sample)
