"""
Tic-tac-toe environment
"""
from collections import defaultdict
import numpy as np
import gymnasium as gym
import tqdm

class TicTacToeEnv(gym.Env):
    """
    Tic-tac-toe environment.

    There are two players, player 1 and player -1.
    - Player 1 places 1, player -1 places -1 on the board.
    - Info tells whose turn it is.

    On each step, the current player place a piece on the board. Then the environment
    - checks for win or draw
    - swaps roles by flipping the board

    The game ends when
    - one player wins (+1 reward)
    - the game is a draw (0 reward)
    - an invalid move is made (-10 reward, game ends)
    """
    # player - symbol mapping
    symbols = {1: "X", -1: "O", 0: "."}

    def __init__(self):
        """
        Initialize the environment.
        Board is 3x3 numpy array with values {-1, 0, 1}.
        Each action is an integer from 0 to 8, representing the position on the board
        """
        super().__init__()
        self.board = np.zeros((3, 3), dtype=np.int8)
        self.current_player = 1   # Player 1 starts the game
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.done = False

    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.

        Args:
            seed (int, optional): Random seed (not used in this deterministic environment).
            options (dict, optional): Additional options (not used).

        Returns:
            observation (np.ndarray): Initial board status.
            info (dict): Additional information (empty).
        """
        super().reset(seed=seed)
        self.board.fill(0)
        self.current_player = 1
        self.done = False
        return self._get_obs(), {"player": self.current_player}

    def step(self, action) -> tuple:
        """
        Step function for the Tick-tack-toe environment.

        Args:
            action (int): An integer from 0 to 8.

        Returns:
            observation (np.ndarray): Next observation of the board.
            reward (float): +1 if win, 0 if draw, -10 if invalid move.
            terminated (bool): True if the game ends.
            truncated (bool): Always False (no step limit).
            info (dict): May indicate "winner", "draw" or "invalid" keys.
        """
        if self.done:
            raise RuntimeError("Cannot step in a finished game.")
        
        # Convert action to row and column
        row, col = divmod(action, 3)  

        # check if the position is already occupied
        if self.board[row, col] != 0:
            return self._get_obs(), -10, True, False, {"invalid": True}
        
        # Place the piece on the board
        self.board[row, col] = self.current_player

        # check for win
        if self._check_win(self.current_player):
            self.done = True
            return self._get_obs(), 1, True, False, {"winner": self.current_player}
        
        # check for draw
        if np.all(self.board != 0):
            self.done = True
            return self._get_obs(), 0, True, False, {"draw": True}
        
        # Neither win nor draw, swap player and continue the game
        self.current_player *= -1
        return self._get_obs(), 0, False, False, {"player": self.current_player}
        
    def _get_obs(self):
        """Get the current observation of the board."""
        return self.board.copy()
    
    def _check_win(self, player):
        """
        Check if the given player has won the game.

        Args:
            player (int): The player to check for a win (+1 or -1).
        Returns:
            bool: True if the player has won, False otherwise.
        """
        b = self.board

        # check rows
        if any([np.all(b[i, :] == player) for i in range(3)]):
            return True
        # check columns
        if any([np.all(b[:, i] == player) for i in range(3)]):
            return True
        # check diagonals
        if np.all(np.diag(b) == player) or np.all(np.diag(np.fliplr(b)) == player):
            return True
        # otherwise, no win
        return False

    def render(self):
        """
        Print the board to the console. 
        - Player 1 is represented by "X"
        - Player -1 is represented by "O"
        """
        for row in self.board:
            print(" ".join(self.symbols[val] for val in row)) # type: ignore
        print()

    def close(self):
        """Close the environment."""
        pass  # No resources to release in this simple environment


def cli_play():
    """
    Play the game on CLI against oneself.
    """
    env = TicTacToeEnv()
    obs, info = env.reset()
    while True:
        env.render()
        player_symbol = env.symbols[info['player']]
        print(f"Player {player_symbol}'s turn")
    
        # Input validation loop
        while True:
            try:
                action = int(input("Enter your move (0-8): "))
                if action < 0 or action > 8:
                    print("Invalid input. Choose between 0 and 8.")
                    continue
                break
            except ValueError:
                print("Invalid input. Enter an integer between 0 and 8.")

        # Step the environment
        obs, reward, done, _, info = env.step(action)
        if done:
            env.render()
            if "winner" in info:
                print(f"Player {env.symbols[info['winner']]} wins!")
            elif "draw" in info:
                print("The game is a draw!")
            else:
                print("Invalid move!")
            break

    # play again?
    if input("Play again? (y/n): ").lower() == 'y':
        cli_play()
    else:
        print("Thanks for playing!")

    
def train_td(num_episodes: int, alpha: float, epsilon: float) -> dict:
    """
    Train a simple agent on the TicTacToe environment with
    Temporal Difference (TD) learning with epsilon-greedy policy.
    The agent is trained by playing against itself.

    Unlike normal game environments, tic-tac-toe involves two players,
    which makes it a bit more complex. In this environment, two steps are
    equal to one step from the perspective of one player, which means we have to
    wait for two steps to update the value function for one player. To handle this,
    we use a buffer that store the trajectory of the last two steps.

    Also, note that the observation space for each player is not the same.
    The first player always sees the board with even number of pieces,
    while the second player sees the board with odd number of pieces.

    Args:
        num_episodes: Number of episodes to train the agent.
        alpha: Leraning rate.
        epsilon: Exploration rate. Choose a random action with this probability.
    Returns:
        dict: A table that maps state to action values.
    """
    # value function: 
    # TODO: should we use action-value function?
    # TODO: we want to use smarter representation of the state than tuple(state.flatten())
    # tuple of the board state to its estimated value
    V = defaultdict(lambda: 0.0)

    # environment
    env = TicTacToeEnv()

    # start training
    for episode in tqdm.tqdm(range(num_episodes), total=num_episodes, desc="Training", dynamic_ncols=True):
        # buffer to store the trajectory of the last two steps
        # each step is a state-action pair
        buffer = []  
        obs, info = env.reset()
        done = False

        while not done:
            # choose action using epsilon-greedy policy
            if np.random.rand() < epsilon:
                # with probability epsilon, choose a random action (exploration)
                valid_actions = get_valid_actions(obs)
                action = np.random.choice(valid_actions)
            else:
                # otherwise, choose the estimated best action (greedy exploitation)
                action = get_best_action(obs, V, info['player'])

            # perform the action and step the environment
            next_obs, reward, done, _, next_info = env.step(action)

            # store the state-action pair in the trajectory
            buffer.append((obs, action))

            # TODO: update the value function using TD learning
            # we use that the reward is 0 for all steps except the last one
            if done:
                # update the value function for the last state in the buffer
                last_state, last_action = buffer[-1]
                V[tuple(last_state.flatten())] += alpha * (reward - V[tuple(last_state.flatten())])
            else:
                # update the value function for the second to last state in the buffer
                second_last_state, second_last_action = buffer[-2]
                V[tuple(second_last_state.flatten())] += alpha * (V[tuple(next_obs.flatten())] - V[tuple(second_last_state.flatten())])
            
            # move to the next state
            obs = next_obs
            info = next_info

    return V


def get_valid_actions(state: np.ndarray) -> list:
    """
    Get a list of valid actions for the current state.
    Valid actions are those positions on the board that are empty (0).

    Args:
        state: Current state of the board.
    Returns:
        list: List of valid actions (integers from 0 to 8).
    """
    # TODO: implement this function
    pass
            

def get_best_action(state: np.ndarray, V: dict, player: int) -> int:
    """
    Find the best action for the given state using the value function V.

    Args:
        state: Current state of the board.
        V: Value function mapping states to action values.
        player: Current player (1 or -1).
    Returns:
        int: The best action to take in the current state.
    """
    values = np.zeros(9)  # value for each action

    # iterate over all valid actions
    for action in get_valid_actions(state):
        next_state = state_transition(state, action, player)
        value = V[tuple(next_state.flatten())]
        values[action] = value
    
    # choose the action with the highest value
    best_action = np.argmax(values)
    return best_action


def state_transition(state: np.ndarray, action: int, player: int) -> np.ndarray:
    """
    Estimate the next state given the current state.
    Used for estimating the value of a state-action pair.
    """
    next_state = state.copy()
    row, col = divmod(action, 3)
    next_state[row, col] = player
    return next_state