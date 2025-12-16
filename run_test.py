import keras
import numpy as np
import pandas as pd
from game import Backend, GameState
from inference import Infer

def check_correct(state1: GameState, state2: GameState) -> bool:
    for x in range(9):
        for y in range(9):
            if state1.board[x][y] == 0:
                return False
            digit = state1.board[x][y]
            digit_check = state2.board[x][y]
            if digit != digit_check:
                return False
    return True

def test(path: str, model) -> float:
    results: float = 0.0
    data = pd.read_csv(path)
    data = pd.DataFrame({"quizzes":data["puzzle"],"solutions":data["solution"]})
    inference = Infer(model)
    for i in range(len(data)):
        print(f"iter: {i}")
        state = GameState()
        state.from_linear(np.array([np.int8(c) for c in data["quizzes"][i]]))
        result, _ = inference.generate_steps(state)
        solved = GameState()
        solved.from_linear(np.array([np.int8(c) for c in data["solutions"][i]]))
        correct = check_correct(result, solved)
        results += (1 if correct else 0)
        print(f"iter: {i} correct?: {'yes' if correct else 'no'}")

    return results / len(data)
    
    

if __name__ == "__main__":
    MODEL_PATH = "best_weights.keras"
    model = keras.models.load_model(MODEL_PATH)
    easy = test("test/data/test_ruste.csv", model)
    medium = test("test/data/test_rustm.csv", model)
    hard = test("test/data/test_rusth.csv", model)
    print("Results:")
    print(f"    Easy (20 holes) accuracy: \033[0;32m{easy:.2f}\033[0;0m")
    print(f"    Medium (40 holes) accuracy: \033[0;32m{medium:.2f}\033[0;0m")
    print(f"    Hard (~60 holes) accuracy: \033[0;32m{hard:.2f}\033[0;0m")

    
