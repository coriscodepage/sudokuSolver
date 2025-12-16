import copy
from typing import Optional
import numpy as np
import pandas as pd
import keras
from keras.optimizers import Adam
from keras.models import Sequential
import keras.layers as kl
from numpy.typing import NDArray

from game import Backend, Coordinates, Frontend, GameState, SetSquareCommand

MODEL_PATH = "best_weights.keras"



class Infer:
    def __init__(self, model: Sequential) -> None:
        self.model: Sequential = model

    @staticmethod
    def norm(a):
        return (a/9)-.5

    @staticmethod
    def denorm(a):
        return (a+.5)*9

    def generate_steps(self, puzzle: GameState, solved: Optional[GameState] = None) -> tuple[GameState, list[SetSquareCommand]]:
        puzzle = copy.copy(puzzle)
        history: list[SetSquareCommand] = []
        feat = Infer.norm(puzzle.board.reshape((9,9,1)))
        pred: NDArray = np.zeros((9, 9), dtype=np.int8)

        while(1):
        
            out = self.model.predict(feat.reshape((1,9,9,1)))  
            out = out.squeeze()

            pred = np.argmax(out, axis=1).reshape((9,9))+1 
            prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 

            feat = Infer.denorm(feat).reshape((9,9))
            mask = (feat==0)

            if(mask.sum()==0):
                break

            prob_new = prob*mask

            ind = np.argmax(prob_new)
            x, y = int(ind//9), int(ind%9)

            val = pred[x][y]
            feat[x][y] = val
            if solved:
                history.append(SetSquareCommand(solved, Coordinates(x, y), np.int8(val)))
            feat = Infer.norm(feat)

        result = GameState()
        result.from_linear(pred)
        return (result, history)

    # def test_accuracy(feats, labels):

    #     correct = 0

    #     for i,feat in enumerate(feats):

    #         pred = inference_sudoku(feat)

    #         true = labels[i].reshape((9,9))+1

    #         if(abs(true - pred).sum()==0):
    #             correct += 1

    #     print(correct/feats.shape[0])

def solve_sudoku():
    model = keras.models.load_model(MODEL_PATH)
    inference = Infer(model) # type: ignore
    backend = Backend()
    backend.generate_sudoku(73)
    result, _ = inference.generate_steps(backend.get_state(), backend.get_solved())
    frontend = Frontend(backend)
    print("Puzzle:")
    frontend.display_board()
    frontend.backend.state = result
    print("Solution:")
    frontend.display_board()
    print("Correct?: " + ("\033[0;32myes" if result.check_correct() else "\033[0;31mno"))

if __name__ == "__main__":
    solve_sudoku()

