import copy
from dataclasses import dataclass
import random
from typing import Optional


@dataclass
class Token:
    value: Optional[int] = None

@dataclass
class GameState:
    board: list[list[Token]]

@dataclass
class Coordinates:
    x: int
    y: int

@dataclass
class CommandResult:
    status: bool
    message: Optional[str]


class Backend:
    def __init__(self):
        self.state: GameState = GameState(board=[[Token() for _ in range(9)] for _ in range(9)])

    def set_square(self, coords: Coordinates, digit: Token):
        self.state.board[coords.x][coords.y] = digit

    def get_state(self) -> GameState:
        return self.state
    
    def find_empty(self, board: list[list[Token]]) -> Optional[Coordinates]:
        for x in range(9):
            for y in range(9):
                if board[x][y].value is None:
                    return Coordinates(x, y)
        return None

    def solve_random(self):
        empty = self.find_empty(self.state.board)
        if not empty:
            return True
            
        nums = list([Token(x) for x in range(1, 10)])
        random.shuffle(nums)
    
        for n in nums:
            if Validator.is_valid_move(self.state, empty, n):
                self.state.board[empty.x][empty.y] = n
                if self.solve_random():
                    return True
                self.state.board[empty.x][empty.y] = Token()
        return False

    def count_solutions(self):
        board = copy.deepcopy(self.state.board)
        solutions = 0
    
        def backtrack():
            nonlocal solutions
            if solutions > 1:
                return
    
            empty = self.find_empty(board)
            if not empty:
                solutions += 1
                return
    
            for n in range(1, 10):
                if Validator.is_valid_move(self.state, empty, Token(n)):
                    board[empty.x][empty.y] = Token(n)
                    backtrack()
                    board[empty.x][empty.y] = Token()
    
        backtrack()
        return solutions    

    def remove_numbers(self, num_holes=40):
        board = copy.deepcopy(self.state.board)
        coords = [(r, c) for r in range(9) for c in range(9)]
        random.shuffle(coords)

        holes = 0
        for r, c in coords:
            if holes >= num_holes:
                break

            backup = board[r][c]
            board[r][c] = Token()

            if self.count_solutions() != 1:
                board[r][c] = backup
            else:
                holes += 1

        return board
    
    def generate_sudoku(self, num_holes=40):
        self.solve_random()
        self.state.board = self.remove_numbers(num_holes)



class Validator:
    
    @staticmethod
    def within_bounds(coords: Coordinates) -> bool:
        return 0 <= coords.x < 9 and 0 <= coords.y < 9
    
    @staticmethod
    def row_valid(state: GameState, coords: Coordinates, digit: Token) -> bool:
        return state.board[coords.x].count(digit) == 0
    
    @staticmethod
    def column_valid(state: GameState, coords: Coordinates, digit: Token) -> bool:
        return all(row[coords.y] != digit for row in state.board)

    @staticmethod
    def box_valid(state: GameState, coords: Coordinates, digit: Token) -> bool:
        x_box = coords.x // 3
        y_box = coords.y // 3
        for x in range(x_box * 3, x_box * 3 + 3):
            for y in range(y_box * 3, y_box * 3 + 3):
                if (state.board[x][y] == digit):
                    return False
        return True

    @staticmethod
    def is_valid_move(state: GameState, coords: Coordinates, digit: Token) -> bool:
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
    def __init__(self, backend: Backend, coords: Coordinates, digit: Token):
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
            print(" | ".join(str(cell.value) if cell.value is not None else " " for cell in row))
            print("-" * (len(row) * 4 - 3))

if __name__ == "__main__":
    backend = Backend()
    frontend = Frontend(backend)
    frontend.display_board()
    print("AAAAAAAAAAAAAA")
    
    #cmd = SetSquareCommand(backend, Coordinates(4, 4), Token(5))
    #cmd.execute()
    frontend.display_board()