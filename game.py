import copy
from dataclasses import dataclass, field
import random
from typing import Optional
from numpy.typing import NDArray
import numpy as np



@dataclass
class Coordinates:
    x: int
    y: int

@dataclass
class Move:
    coords: Coordinates
    digit: np.int8

@dataclass
class GameState:
    board: NDArray[np.int8] = field(default_factory=lambda: np.zeros((9, 9), dtype=np.int8))

    def check_solved(self) -> bool:
        return all(self.board[x][y] != 0 for x in range(9) for y in range(9))
    
    def check_correct(self) -> bool:
        for x in range(9):
            for y in range(9):
                if self.board[x][y] == 0:
                    return False
                digit = self.board[x][y]
                self.board[x][y] = 0
                if not Validator.is_valid_move(self, Coordinates(x, y), digit):
                    return False
                self.board[x][y] = digit
        return True

    def is_finished(self):
        return GameState.check_solved(self)
    
    def available_actions(self) -> list[Move]:
        actions = []
        for x in range(9):
            for y in range(9):
                if self.board[x][y] != 0:
                    continue
                
                used = set(self.board[x]) | set(self.board[i][y] for i in range(9))
                bx, by = (x//3)*3, (y//3)*3
                for i in range(bx, bx+3):
                    for j in range(by, by+3):
                        used.add(self.board[i][j])
                used.discard(None)
                for d in np.arange(1, 10):
                    if d not in used:
                        actions.append(Move(Coordinates(x, y), d))
        return actions
    
    def from_linear(self, array: NDArray[np.int8]):
        self.board = array.reshape((9,9)).astype(np.int8)

    @staticmethod
    def find_empty(board: NDArray[np.int8]) -> Optional[Coordinates]:
        for x in range(9):
            for y in range(9):
                if board[x][y] == 0:
                    return Coordinates(x, y)
        return None

class Backend:
    def __init__(self):
        self.state: GameState = GameState()
        self.solved: GameState = GameState()

    def get_state(self) -> GameState:
        return self.state

    def solve_random(self):
        empty = GameState.find_empty(self.state.board)
        if not empty:
            return True
            
        nums = np.arange(1,10)
        np.random.shuffle(nums)
    
        for n in nums:
            if Validator.is_valid_move(self.state, empty, n):
                self.state.board[empty.x][empty.y] = n
                if self.solve_random():
                    return True
                self.state.board[empty.x][empty.y] = 0
        return False

    def count_solutions(self, board: NDArray[np.int8]):
        board = board.copy()
        solutions = 0
        
        temp_state = GameState()

        def backtrack():
            nonlocal solutions, temp_state
            if solutions > 1:
                return

            empty = GameState.find_empty(board)
            if not empty:
                solutions += 1
                return

            temp_state.board = board
            for n in np.arange(1, 10):

                if Validator.is_valid_move(temp_state, empty, n):
                    board[empty.x][empty.y] = n
                    backtrack()
                    board[empty.x][empty.y] = 0

        backtrack()
        return solutions

    def remove_numbers(self, num_holes: int = 40):
        board = self.state.board.copy()
        coords = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(coords)

        holes = 0
        for r, c in coords:
            if holes >= num_holes:
                break

            backup = board[r][c]
            board[r][c] = 0

            if self.count_solutions(board) != 1:
                board[r][c] = backup
            else:
                holes += 1

        return board
    
    def generate_sudoku(self, num_holes: int = 40):
        self.solve_random()
        self.solved.board = self.state.board.copy()
        self.state.board = self.remove_numbers(num_holes)

    def generate_from_solved(self,  num_holes: int = 40):
        self.state.board = self.remove_numbers(num_holes)

    def get_solved(self) -> GameState:
        return self.solved

class Validator:
    
    @staticmethod
    def within_bounds(coords: Coordinates) -> bool:
        return 0 <= coords.x < 9 and 0 <= coords.y < 9
    
    @staticmethod
    def row_valid(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        row = state.board[coords.x]
        for v in row:
            if v == digit:
                return False
        return True
    
    @staticmethod
    def column_valid(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        for i in range(9):
            if state.board[i, coords.y] == digit:
                return False
        return True
    
    @staticmethod
    def box_valid(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        x_start = (coords.x // 3) * 3
        y_start = (coords.y // 3) * 3
        
        for i in range(3):
            x = x_start + i
            if (state.board[x, y_start] == digit or 
                state.board[x, y_start + 1] == digit or 
                state.board[x, y_start + 2] == digit):
                return False
        return True
    
    @staticmethod
    def is_valid_move(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        if not (0 <= coords.x < 9 and 0 <= coords.y < 9):
            return False
        
        row = state.board[coords.x]
        for v in row:
            if v == digit:
                return False
        
        for i in range(9):
            if state.board[i, coords.y] == digit:
                return False
        
        x_start = (coords.x // 3) * 3
        y_start = (coords.y // 3) * 3
        for i in range(3):
            x = x_start + i
            if (state.board[x, y_start] == digit or 
                state.board[x, y_start + 1] == digit or 
                state.board[x, y_start + 2] == digit):
                return False
        
        return True


class SetSquareCommand:
    def __init__(self, solved: GameState, coords: Coordinates, digit: np.int8):
        self.coords = coords
        self.digit = digit
        self.solved = solved
        self.valid: bool = (self.solved.board[self.coords.x][self.coords.y] == digit)

class Frontend:
    def __init__(self, backend: Backend):
        self.backend = backend

    def display_board(self):
        state = self.backend.get_state()
        for row in state.board:
            print(" "*4, end="")
            print(" | ".join(str(cell) if cell != 0 else " " for cell in row))
            print(" "*4, end="")
            print("-" * (len(row) * 4 - 3))

if __name__ == "__main__":
    backend = Backend()
    frontend = Frontend(backend)
    backend.solve_random()
    backend.generate_sudoku(60)
    frontend.display_board()