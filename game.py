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

    @staticmethod
    def find_empty(board: NDArray[np.int8]) -> Optional[Coordinates]:
        for x in range(9):
            for y in range(9):
                if board[x][y] == 0:
                    return Coordinates(x, y)
        return None

@dataclass
class CommandResult:
    status: bool
    message: Optional[str]


class Backend:
    def __init__(self):
        self.state: GameState = GameState()

    def set_square(self, coords: Coordinates, digit: np.int8):
        self.state.board[coords.x][coords.y] = digit

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

    def count_solutions(self):
        board = copy.deepcopy(self.state.board)
        solutions = 0
    
        def backtrack():
            nonlocal solutions
            if solutions > 1:
                return
    
            empty = GameState.find_empty(board)
            if not empty:
                solutions += 1
                return
    
            for n in np.arange(1, 10):
                if Validator.is_valid_move(self.state, empty, n):
                    board[empty.x][empty.y] = n
                    backtrack()
                    board[empty.x][empty.y] = 0
    
        backtrack()
        return solutions    

    def remove_numbers(self, num_holes: int = 40):
        board = copy.deepcopy(self.state.board)
        coords = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(coords)

        holes = 0
        for r, c in coords:
            if holes >= num_holes:
                break

            backup = board[r][c]
            board[r][c] = 0

            if self.count_solutions() != 1:
                board[r][c] = backup
            else:
                holes += 1

        return board
    
    def generate_sudoku(self, num_holes: int = 40):
        self.solve_random()
        self.state.board = self.remove_numbers(num_holes)



class Validator:
    
    @staticmethod
    def within_bounds(coords: Coordinates) -> bool:
        return 0 <= coords.x < 9 and 0 <= coords.y < 9
    
    @staticmethod
    def row_valid(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        for v in state.board[coords.x]:
            if(v == digit):
                return False
        return True
    
    @staticmethod
    def column_valid(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        for row in state.board:
            if(row[coords.y] == digit):
                return False
        return True

    @staticmethod
    def box_valid(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        x_box = coords.x // 3
        y_box = coords.y // 3
        for x in range(x_box * 3, x_box * 3 + 3):
            for y in range(y_box * 3, y_box * 3 + 3):
                if (state.board[x][y] == digit):
                    return False
        return True

    @staticmethod
    def is_valid_move(state: GameState, coords: Coordinates, digit: np.int8) -> bool:
        if (not Validator.within_bounds(coords)):
            return False
        
        if (not Validator.column_valid(state, coords, digit)):
            return False
        
        if (not Validator.row_valid(state, coords, digit)):
            return False
        
        if (not Validator.box_valid(state, coords, digit)):
            return False
        
        return True


class SetSquareCommand:
    def __init__(self, backend: Backend, coords: Coordinates, digit: np.int8):
        self.coords = coords
        self.digit = digit
        self.backend = backend

    def execute(self) -> CommandResult:
        if not Validator.is_valid_move(self.backend.get_state(), self.coords, self.digit):
            return CommandResult(status=False, message="Invalid move")
        self.backend.set_square(self.coords, self.digit)
        return CommandResult(status=True, message=None)

class Frontend:
    def __init__(self, backend: Backend):
        self.backend = backend

    def display_board(self):
        state = self.backend.get_state()
        for row in state.board:
            print(" | ".join(str(cell) if cell is not None else " " for cell in row))
            print("-" * (len(row) * 4 - 3))

if __name__ == "__main__":
    backend = Backend()
    frontend = Frontend(backend)
    backend.solve_random()
    frontend.display_board()