import numpy as np


class Connect4Env:
    def __init__(self):
        self.rows = 6
        self.columns = 7
        self.board = self._initialize_board()
        self.done = False
        self.current_player = 1

    def _initialize_board(self):
        return np.zeros((self.rows, self.columns), dtype=int)

    def reset(self):
        self.board = self._initialize_board()
        self.done = False
        self.current_player = 1
        return self.board

    def valid_actions(self):
        return [c for c in range(self.columns) if self.board[0, c] == 0]

    def step(self, action):
        if action not in self.valid_actions():
            raise ValueError("Invalid action")

        for row in reversed(range(self.rows)):
            if self.board[row, action] == 0:
                self.board[row, action] = self.current_player
                break

        if self._check_win(self.current_player):
            self.done = True
            reward = 1
        elif len(self.valid_actions()) == 0:
            self.done = True
            reward = 0
        else:
            reward = 0

        self.current_player *= -1
        reward = np.clip(reward, -1, 1)
        return self.board.copy(), reward, self.done

    def _check_win(self, player):
        for row in range(self.rows):
            for col in range(self.columns - 3):
                if np.all(self.board[row, col:col + 4] == player):
                    return True
        for row in range(self.rows - 3):
            for col in range(self.columns):
                if np.all(self.board[row:row + 4, col] == player):
                    return True
        for row in range(self.rows - 3):
            for col in range(self.columns - 3):
                if np.all([self.board[row + i, col + i] == player for i in range(4)]):
                    return True
                if np.all([self.board[row + 3 - i, col + i] == player for i in range(4)]):
                    return True
        return False
