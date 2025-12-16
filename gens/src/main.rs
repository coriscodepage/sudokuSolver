use std::{fmt::Display, fs::File, io::{BufWriter, Write}};

use rand::seq::SliceRandom;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

#[derive(Debug, Clone, Copy)]
pub struct Coordinates {
    pub x: u8,
    pub y: u8,
}

impl Coordinates {
    pub fn new(x: u8, y: u8) -> Self {
        Self { x, y }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Pair {
    pub question: Puzzle,
    pub answer: Puzzle,
}

impl Pair {
    pub fn new(question: Puzzle, answer: Puzzle) -> Self {
        Self { question, answer }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Puzzle {
    pub board: [[u8; 9]; 9],
}

impl Display for Puzzle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
       for row in self.board {
        for num in row {
            write!(f, "{}", num)?;
        }
       }
       Ok(())
    }
}

impl Puzzle {
    pub fn new() -> Self {
        Self { board: [[0; 9]; 9] }
    }

    pub fn check_solved(&self) -> bool {
        self.board.iter().all(|row| row.iter().all(|&cell| cell != 0))
    }

    pub fn check_correct(&mut self) -> bool {
        for x in 0usize..9 {
            for y in 0usize..9 {
                if self.board[x][y] == 0 {
                    return false;
                }
                let digit = self.board[x][y];
                self.board[x][y] = 0;
                if Validator::is_valid_move(self, Coordinates::new(x as u8, y as u8), digit) {
                    return false;
                }
                self.board[x][y] = digit;
            }
        }
        return true;
    }

    pub fn find_empty(&self) -> Option<Coordinates> {
        for x in 0usize..9 {
            for y in 0usize..9 {
                if self.board[x][y] == 0 {
                    return Some(Coordinates::new(x as u8, y as u8));
                }
            }
        }
        None
    }

}

pub struct Validator;

impl Validator {
    #[inline]
    pub fn within_bounds(coords: Coordinates) -> bool {
        coords.x < 9 && coords.y < 9
    }

    #[inline]
    pub fn row_valid(puzzle: &Puzzle, coords: Coordinates, digit: u8) -> bool {
        !puzzle.board[coords.x as usize].contains(&digit)
    }

    #[inline]
    pub fn column_valid(puzzle: &Puzzle, coords: Coordinates, digit: u8) -> bool {
        !(0..9).any(|i| puzzle.board[i][coords.y as usize] == digit)
    }

    #[inline]
    pub fn box_valid(puzzle: &Puzzle, coords: Coordinates, digit: u8) -> bool {
        let x_start = (coords.x as usize / 3) * 3;
        let y_start = (coords.y as usize / 3) * 3;

        for i in 0usize..3 {
            let x = x_start + i;
            if puzzle.board[x][y_start] == digit || puzzle.board[x][y_start + 1] == digit || puzzle.board[x][y_start + 2] == digit {
                return false;
            }
        }

        return true;
    }

    #[inline]
    pub fn is_valid_move(puzzle: &Puzzle, coords: Coordinates, digit: u8) -> bool {
        Validator::within_bounds(coords)
            && Validator::row_valid(puzzle, coords, digit)
            && Validator::column_valid(puzzle, coords, digit)
            && Validator::box_valid(puzzle, coords, digit)
    }

}

pub struct Generate;

impl Generate {
    pub fn solve_random(puzzle: &mut Puzzle) -> Option<&mut Puzzle> {
        let empty_space = match puzzle.find_empty() {
            Some(s) => s,
            None => return Some(puzzle),
        };
        let mut nums: Vec<u8> = (1..=9).collect();
        nums.shuffle(&mut rand::rng());
        for n in nums {
            if Validator::is_valid_move(puzzle, empty_space, n) {
                puzzle.board[empty_space.x as usize][empty_space.y as usize] = n;
                if Generate::solve_random(puzzle).is_some() {
                    return Some(puzzle);
                }
                puzzle.board[empty_space.x as usize][empty_space.y as usize] = 0;
            }
        }
        None
    }

    pub fn count_solutions(mut puzzle: Puzzle) -> usize {
        let mut solutions = 0;
        

        fn backtrack(puzzle: &mut Puzzle, solutions: &mut usize) {
            if *solutions > 1 {
                return;
            }
            let empty_space = match puzzle.find_empty() {
                Some(s) => s,
                None => {
                    *solutions += 1; 
                    return;
                }
            };
            for n in 1..=9 {
                if Validator::is_valid_move(puzzle, empty_space, n) {
                    puzzle.board[empty_space.x as usize][empty_space.y as usize] = n;
                    backtrack(puzzle, solutions);
                    puzzle.board[empty_space.x as usize][empty_space.y as usize] = 0;
                }
            }
        }
        backtrack(&mut puzzle, &mut solutions);
        solutions
    }

    pub fn remove_numbers(puzzle: &mut Puzzle, num_holes: usize) {
        let mut coords: Vec<(usize, usize)> = (0..9)
            .flat_map(|r| (0..9).map(move |c| (r, c)))
            .collect();
        
        coords.shuffle(&mut rand::rng());
        
        let mut holes = 0;
        for (r, c) in coords {
            if holes >= num_holes {
                break;
            }

            let backup = puzzle.board[r][c];
            puzzle.board[r][c] = 0;

            if Generate::count_solutions(*puzzle) != 1 {
                puzzle.board[r][c] = backup;
            } else {
                holes += 1;
            }
        }
    }
    
}

#[allow(dead_code)]
fn display_board(puzzle: &Puzzle) {
    let board = puzzle.board;
    for row in board {
        println!("{:?}", row)
    }
}

#[allow(dead_code)]
fn generate_puzzle(count: usize) -> Vec<Pair> {
    (0..count)
        .into_par_iter()
        .map(|_| {
            let mut puzzle = Puzzle::new();
            Generate::solve_random(&mut puzzle);
            let solution = puzzle;
            Generate::remove_numbers(&mut puzzle, 60);
            Pair::new(puzzle, solution)
        })
        .collect()
}


fn generate_puzzles(count: usize) -> Vec<Pair> {
    let mut results: Vec<Pair> = (0..(count as f32 *0.4) as usize)
        .into_par_iter()
        .map(|_| {
            let mut puzzle = Puzzle::new();
            Generate::solve_random(&mut puzzle);
            let solution = puzzle;
            Generate::remove_numbers(&mut puzzle, 40);
            Pair::new(puzzle, solution)
        })
        .collect();
    
    let mut easy: Vec<Pair> = (0..(count as f32 *0.3) as usize)
        .into_par_iter()
        .map(|_| {
            let mut puzzle = Puzzle::new();
            Generate::solve_random(&mut puzzle);
            let solution = puzzle;
            Generate::remove_numbers(&mut puzzle, 20);
            Pair::new(puzzle, solution)
        })
        .collect();
    results.append(&mut easy);

    let mut hard: Vec<Pair> = (0..(count as f32 *0.3) as usize)
        .into_par_iter()
        .map(|_| {
            let mut puzzle = Puzzle::new();
            Generate::solve_random(&mut puzzle);
            let solution = puzzle;
            Generate::remove_numbers(&mut puzzle, 50);
            Pair::new(puzzle, solution)
        })
        .collect();
    results.append(&mut hard);

    results
}

fn main() {
    let all_results = generate_puzzles(1e6 as usize * 5);
    let file = File::create("data_rust.csv").unwrap();
    let mut file = BufWriter::new(file);
    writeln!(file, "puzzle,solution").unwrap();
    for pair in all_results {
        writeln!(file, "{},{}", pair.question, pair.answer).unwrap();
    }
}
