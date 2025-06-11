"""
k-armed bandit problem environment and a epsilon-greedy agent.
"""
import numpy as np
import gymnasium as gym


class BanditEnv(gym.Env):
    """
    k-armed bandit environment.
    """
    def __init__(self, k: int = 10):
        super().__init__()
        self.k = k
        self.action_space = gym.spaces.Discrete(k)
        self.observation_space = gym.spaces.Discrete(1)  # Single state

        # Initialize the true action values
        self.q = np.random.normal(0, 1, k)
        # best action
        self.best_action = np.argmax(self.q)

    def reset(self):
        """
        Reset the environment. Randomly initialize the true action values.
        Returns:
            observation (None): There is no observation in this problem.
            info (dict): Additional information (empty).
        """
        self.q = np.random.normal(0, 1, self.k)
        self.best_action = np.argmax(self.q)
        return None, {}

    def step(self, action: int) -> tuple:
        """
        Step the environment by taking an action.
        Args:
            action (int): The action to take (0 to k-1).
        Returns:
            observation (None): There is no observation in this problem.
            reward (float): The reward received for the action.
            done (bool): Whether the episode is done, always False.
            info (dict): Additional information. "is_optimal" Indicates if the action was optimal.
        """
        reward = np.random.normal(self.q_true[action], 1)
        return None, reward, False, {"is_optimal": action == self.best_action}