"""
Tick-tack-toe environment
"""
import numpy as np
import gymnasium as gym


class TicTacToeEnv(gym.Env):
    """
    Tick-tack-toe environment.

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