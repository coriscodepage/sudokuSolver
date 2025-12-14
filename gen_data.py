from multiprocessing import Pool
from game import Backend
from itertools import chain
import copy
import cProfile
import pandas as pd
import numpy as np

def create(num: int):
    results = []
    backend = Backend()
    for _ in range(num):
        backend.solve_random()
        solved = backend.state.board.copy()
        backend.generate_from_solved()
        puzzle = backend.state.board.copy()
        results.append((puzzle, solved))
    return results

def run():
    nt = 12
    with Pool(nt) as pool:
        all_results = pool.map(create, [int(1e6 * 2 / nt)] * nt)
    
    flat = list(chain.from_iterable(all_results))
    questions, answers = zip(*flat)
    
    questions_str = [''.join(map(str, q.flatten())) for q in questions]
    answers_str = [''.join(map(str, a.flatten())) for a in answers]
    
    df = pd.DataFrame({'puzzle': questions_str, 'solution': answers_str})
    df.to_csv('data.csv', index=False)

run()
#cProfile.run('create(100)', sort='ncalls')