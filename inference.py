import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import Sequence
import keras.layers as kl
from numpy.typing import NDArray

from game import Backend, Frontend, GameState

model = Sequential()

model.add(kl.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
model.add(kl.BatchNormalization())
model.add(kl.Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(kl.BatchNormalization())
model.add(kl.Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

model.add(kl.Flatten())
model.add(kl.Dense(81*9))
model.add(kl.Reshape((-1, 9)))
model.add(kl.Activation('softmax'))

adam = keras.optimizers.Adam(learning_rate=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy']) # type: ignore

model.load_weights('best_weights.keras')

def norm(a):
    return (a/9)-.5

def denorm(a):
    return (a+.5)*9

def inference_sudoku(sample):
    
    feat = sample
    pred = None
    
    while(1):
    
        out = model.predict(feat.reshape((1,9,9,1)))  
        out = out.squeeze()

        pred = np.argmax(out, axis=1).reshape((9,9))+1 
        prob = np.around(np.max(out, axis=1).reshape((9,9)), 2) 
        
        feat = denorm(feat).reshape((9,9))
        mask = (feat==0)
     
        if(mask.sum()==0):
            break
            
        prob_new = prob*mask
    
        ind = np.argmax(prob_new)
        x, y = (ind//9), (ind%9)

        val = pred[x][y]
        feat[x][y] = val
        feat = norm(feat)
    
    return pred

def test_accuracy(feats, labels):
    
    correct = 0
    
    for i,feat in enumerate(feats):
        
        pred = inference_sudoku(feat)
        
        true = labels[i].reshape((9,9))+1
        
        if(abs(true - pred).sum()==0):
            correct += 1
        
    print(correct/feats.shape[0])

def solve_sudoku():
    backend = Backend()
    backend.generate_sudoku()

    game = backend.state.board.reshape((9,9,1))
    game = norm(game)
    game = inference_sudoku(game)
    frontend = Frontend(backend)
    print("Puzzle:")
    frontend.display_board()
    result = GameState()
    result.from_linear(game) # type: ignore
    frontend.backend.state = result
    print("Solution:")
    frontend.display_board()
    print("Correct?: " + "yes" if result.check_solved() else "no")


game = solve_sudoku()

