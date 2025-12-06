
from dataclasses import dataclass, field
import copy
import random
from typing import Optional
import numpy
from game import Backend, GameState, Move
import cProfile
import numpy as np

@dataclass
class MCTSNode:
    state: GameState
    parent: Optional["MCTSNode"] = None
    action: Optional[Move] = None
    children: list["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    wins: float = 0.0
    _untried_actions: Optional[list[Move]] = field(default=None, repr=False)

    @property
    def untried_actions(self):
        if self._untried_actions is None:
            self._untried_actions = self.state.available_actions()
        return self._untried_actions

    def expand(self) -> Optional["MCTSNode"]:
        action = self.untried_actions.pop()
        appended_state = GameState()
        appended_state.board = np.copy(self.state.board)
        appended_state.board[action.coords.x][action.coords.y] = action.digit
        if not MCTSNode.propagate(appended_state):
            return None
        child = MCTSNode(appended_state, self, action)
        self.children.append(child)
        return child
    
    def best_child(self, c: float = 1.4) -> "MCTSNode":
        best_child = self.children[0]
        best_value = -float('inf')
        log_parent = numpy.log(self.visits)
        
        for child in self.children:
            if child.visits == 0:
                uct_value = float('inf')
            else:
                exploit = child.wins / child.visits
                explore = c * numpy.sqrt(log_parent / child.visits)
                uct_value = exploit + explore
            
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        
        return best_child
    
    def rollout(self) -> float:        
        state = copy.copy(self.state)
        state.board = np.copy(self.state.board)

        if not MCTSNode.propagate(state):
            return 0.0
        
        while True:
            if state.check_solved():
                return 1.0
            
            acts = state.available_actions()
            
            if not acts:
                return 0.0
                        
            cell_cands = {}
            for move in acts:
                key = (move.coords.x, move.coords.y)
                cell_cands.setdefault(key, set()).add(move.digit)
            
            min_cell_key = min(cell_cands.items(), key=lambda kv: len(kv[1]))[0]
            
            choices = [move for move in acts if (move.coords.x, move.coords.y) == min_cell_key]
            
            action = random.choice(choices)
            
            state.board[action.coords.x][action.coords.y] = action.digit
            MCTSNode.propagate(state)
            
            if state is None:
                return 0.0

    def backpropagate(self, winning: float):
        self.visits += 1

        self.wins += winning
        
        if self.parent:
            self.parent.backpropagate(winning)
    
    @staticmethod
    def propagate(state) -> Optional[GameState]:
        changed = True
        while changed:
            changed = False
            for x in range(9):
                for y in range(9):
                    if state.board[x][y] != 0:
                        continue
                    
                    used = set(state.board[x])
                    for i in range(9):
                        used.add(state.board[i][y])
                    bx, by = (x//3)*3, (y//3)*3
                    for i in range(bx, bx+3):
                        for j in range(by, by+3):
                            used.add(state.board[i][j])
                    candidates = [d for d in range(1, 10) if d not in used]

                    if len(candidates) == 0:
                        return None
                    if len(candidates) == 1:
                        state.board[x][y] = candidates[0]
                        changed = True
        return state


    def mcts(self, iterations: int = 500) -> Optional["MCTSNode"]:
        for _ in range(iterations):
            node = self

            while not node.state.is_finished() and len(node.untried_actions) == 0 and len(node.children) > 0:
                node = node.best_child()

            if not node.state.is_finished() and len(node.untried_actions) > 0:
                child_node = node.expand()
                if child_node is None:
                    reward = 0.0
                else:
                    node = child_node
                    reward = node.rollout()
            else:
                reward = node.rollout()

            node.backpropagate(reward)

        if not self.children:
            return None
        return max(self.children, key=lambda c: c.visits)


def solve_with_mcts(puzzle, iter_per_move=2000, max_moves=100):
    board = GameState()
    for x in range(9):
        for y in range(9):
            board.board[x][y] = puzzle[x][y]
    root = MCTSNode(board)
    for step in range(max_moves):
        if board.check_solved():
            return board, True
        res = root.mcts(iter_per_move)
        if res is None:
            return board, False
        board = res.state
        root = res
    return board, board.check_solved()

if __name__ == "__main__":
    puzzle = [
        [5,3,0,0,7,0,0,0,0],
        [6,0,0,1,9,5,0,0,0],
        [0,9,8,0,0,0,0,6,0],
        [8,0,0,0,6,0,0,0,3],
        [4,0,0,8,0,3,0,0,1],
        [7,0,0,0,2,0,0,0,6],
        [0,6,0,0,0,0,2,8,0],
        [0,0,0,4,1,9,0,0,5],
        [0,0,0,0,8,0,0,7,9],
    ]
    ba = Backend()
    ba.generate_sudoku()
    puzzle = ba.get_state().board
    #cProfile.run('solve_with_mcts(puzzle)', sort='cumtime')
    solved_board, ok = solve_with_mcts(puzzle, iter_per_move=2000, max_moves=81)
    print("Solved:", ok)
    for row in solved_board.board:
        print(" | ".join(str(cell) if cell is not None else " " for cell in row))
        print("-" * (len(row) * 4 - 3))