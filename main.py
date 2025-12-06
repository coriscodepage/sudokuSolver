import math, random, copy
from collections import defaultdict
import cProfile

# -----------------------
# Minimal Sudoku utilities
# -----------------------
def legal_actions(board):
    """Return list of (r,c,d) legal actions for empty cells."""
    acts = []
    for r in range(9):
        for c in range(9):
            if board[r][c] != 0:
                continue
            used = set(board[r]) | set(board[i][c] for i in range(9))
            br, bc = (r//3)*3, (c//3)*3
            for i in range(br, br+3):
                for j in range(bc, bc+3):
                    used.add(board[i][j])
            for d in range(1,10):
                if d not in used:
                    acts.append((r,c,d))
    return acts

def is_solved(board):
    for r in range(9):
        if 0 in board[r]:
            return False
    return True

def apply_action(board, action):
    r,c,d = action
    nb = [row[:] for row in board]
    nb[r][c] = d
    return nb

# Simple propagation: fill naked singles repeatedly
def propagate(board):
    b = [row[:] for row in board]
    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if b[r][c] != 0:
                    continue
                used = set(b[r]) | set(b[i][c] for i in range(9))
                br, bc = (r//3)*3, (c//3)*3
                for i in range(br, br+3):
                    for j in range(bc, bc+3):
                        used.add(b[i][j])
                candidates = [d for d in range(1,10) if d not in used]
                if len(candidates) == 0:
                    return None  # contradiction
                if len(candidates) == 1:
                    b[r][c] = candidates[0]
                    changed = True
    return b

# -----------------------
# MCTS node
# -----------------------
class Node:
    def __init__(self, board, parent=None, action_from_parent=None):
        self.board = board
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.children = {}  # action -> Node
        self.untried_actions = legal_actions(board)
        self.N = 0  # visits
        self.W = 0.0  # total reward

# -----------------------
# MCTS core
# -----------------------
def uct_select(node, c=1.4):
    # choose child maximizing UCT value
    best = None
    best_val = -1e9
    for a, child in node.children.items():
        if child.N == 0:
            uct = float('inf')
        else:
            uct = child.W/child.N + c * math.sqrt(math.log(node.N) / child.N)
        if uct > best_val:
            best_val = uct
            best = child
    return best

def rollout_policy(board):
    # rollout: propagate singles then random legal moves until terminal or contradiction
    b = propagate(board)
    if b is None:
        return 0.0
    while True:
        if is_solved(b):
            # optionally verify full correctness (should hold if we never placed illegal moves)
            return 1.0
        acts = legal_actions(b)
        if not acts:
            return 0.0
        # prefer MRV: choose action on cell with fewest candidates
        # compute candidate counts per empty cell
        cell_cands = {}
        for (r,c,d) in acts:
            cell_cands.setdefault((r,c), set()).add(d)
        # pick cell with minimal candidate count
        min_cell = min(cell_cands.items(), key=lambda kv: len(kv[1]))[0]
        choices = [(r,c,d) for (r,c,d) in acts if (r,c)==min_cell]
        a = random.choice(choices)
        b = apply_action(b, a)
        b = propagate(b)
        if b is None:
            return 0.0

def mcts(root_board, iter_limit=2000):
    root = Node(root_board)
    for it in range(iter_limit):
        node = root
        # --- selection
        while node.untried_actions == [] and node.children:
            node = uct_select(node)
        # --- expansion
        if node.untried_actions:
            a = node.untried_actions.pop(random.randrange(len(node.untried_actions)))
            new_board = apply_action(node.board, a)
            new_board = propagate(new_board)
            if new_board is None:
                reward = 0.0
                # backpropagate immediate failure
                child = Node(root_board, parent=node, action_from_parent=a)
                child.N = 1
                child.W = 0.0
                node.children[a] = child
                node = child
                # backprop below will handle
            else:
                child = Node(new_board, parent=node, action_from_parent=a)
                node.children[a] = child
                node = child
                # --- rollout
                reward = rollout_policy(node.board)
        else:
            # terminal or fully expanded leaf
            if node.board is None:
                reward = 0.0
            else:
                reward = rollout_policy(node.board)
        # --- backpropagate
        while node is not None:
            node.N += 1
            node.W += reward
            node = node.parent
    # choose best child by visit count
    if not root.children:
        return None
    best_action = max(root.children.items(), key=lambda kv: kv[1].N)[0]
    return best_action, root.children[best_action]

# -----------------------
# Example usage: pick moves until solved
# -----------------------
def solve_with_mcts(puzzle, iter_per_move=2000, max_moves=100):
    board = [row[:] for row in puzzle]
    for step in range(max_moves):
        if is_solved(board):
            return board, True
        res = mcts(board, iter_limit=iter_per_move)
        if res is None:
            return board, False
        action, child = res
        board = child.board  # apply the chosen child state (already propagated)
    return board, is_solved(board)

# -----------------------
# simple test (use a real puzzle here)
# -----------------------
if __name__ == "__main__":
    # trivial test board (you should pass a real puzzle)
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
    cProfile.run('solve_with_mcts(puzzle)', sort='cumtime')
    #solved_board, ok = solve_with_mcts(puzzle)
    #print("Solved:", ok)
    #for r in solved_board:
    #    print(r)
