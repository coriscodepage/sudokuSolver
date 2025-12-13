import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import random
from dataclasses import dataclass
import time

"""
SELF-SUPERVISED SUDOKU LEARNING

Approach: Train the network to predict WHICH CELLS WILL BE EASY TO FILL
based on the current board state.

Key insight: Instead of learning to solve Sudoku directly, learn to recognize
which positions will lead to progress. The model learns from its own experience
by trying different moves and seeing which ones lead to solutions.

This is inspired by AlphaZero but adapted for constraint satisfaction.
"""

@dataclass
class TrainingExample:
    state: np.ndarray
    target_cell: tuple  # (x, y) - which cell to fill next
    target_digit: int   # What digit to place
    outcome: float      # Did this path lead to success?

class SudokuEnvironment:
    """Environment for self-play"""
    
    def __init__(self, puzzle):
        self.initial_board = puzzle.copy()
        self.board = puzzle.copy()
        self.history = []
    
    def reset(self):
        """Reset to initial state"""
        self.board = self.initial_board.copy()
        self.history = []
    
    def get_valid_moves(self):
        """Get all valid moves (cell, digit pairs)"""
        moves = []
        
        for x in range(9):
            for y in range(9):
                if self.board[x][y] != 0:
                    continue
                
                # Find valid digits for this cell
                used = set(self.board[x, :]) | set(self.board[:, y])
                bx, by = (x // 3) * 3, (y // 3) * 3
                used |= set(self.board[bx:bx+3, by:by+3].flatten())
                used.discard(0)
                
                for digit in range(1, 10):
                    if digit not in used:
                        moves.append((x, y, digit))
        
        return moves
    
    def get_cells_with_fewest_candidates(self):
        """Get cells with minimum number of valid candidates"""
        cell_candidates = {}
        
        for x in range(9):
            for y in range(9):
                if self.board[x][y] != 0:
                    continue
                
                used = set(self.board[x, :]) | set(self.board[:, y])
                bx, by = (x // 3) * 3, (y // 3) * 3
                used |= set(self.board[bx:bx+3, by:by+3].flatten())
                used.discard(0)
                
                candidates = [d for d in range(1, 10) if d not in used]
                if candidates:
                    cell_candidates[(x, y)] = candidates
        
        if not cell_candidates:
            return []
        
        min_count = min(len(c) for c in cell_candidates.values())
        
        # Return cells with min candidates and their possible digits
        strategic_moves = []
        for (x, y), cands in cell_candidates.items():
            if len(cands) == min_count:
                for digit in cands:
                    strategic_moves.append((x, y, digit))
        
        return strategic_moves
    
    def make_move(self, x, y, digit):
        """Make a move and do constraint propagation"""
        self.board[x][y] = digit
        self.history.append((x, y, digit))
        self._propagate()
    
    def _propagate(self):
        """Simple constraint propagation - naked singles only"""
        changed = True
        iterations = 0
        
        while changed and iterations < 3:
            changed = False
            iterations += 1
            
            for x in range(9):
                for y in range(9):
                    if self.board[x][y] != 0:
                        continue
                    
                    used = set(self.board[x, :]) | set(self.board[:, y])
                    bx, by = (x // 3) * 3, (y // 3) * 3
                    used |= set(self.board[bx:bx+3, by:by+3].flatten())
                    used.discard(0)
                    
                    candidates = [d for d in range(1, 10) if d not in used]
                    
                    if len(candidates) == 1:
                        self.board[x][y] = candidates[0]
                        changed = True
    
    def is_solved(self):
        """Check if solved"""
        return np.count_nonzero(self.board) == 81
    
    def is_valid(self):
        """Check if board is still valid (no contradictions)"""
        # Quick check: any cell with no candidates?
        for x in range(9):
            for y in range(9):
                if self.board[x][y] != 0:
                    continue
                
                used = set(self.board[x, :]) | set(self.board[:, y])
                bx, by = (x // 3) * 3, (y // 3) * 3
                used |= set(self.board[bx:bx+3, by:by+3].flatten())
                used.discard(0)
                
                if len(used) >= 9:  # No valid digit possible
                    return False
        
        return True


class PolicyValueNetwork:
    """Network that learns from self-play"""
    
    def __init__(self, model):
        self.model = model
    
    def predict(self, board):
        """Predict policy and value for a board state"""
        board_input = board.reshape(1, 9, 9, 1).astype(np.float32)
        policy_logits, value = self.model.predict(board_input, verbose=0)
        return policy_logits[0], float(value[0][0])
    
    def get_move_probabilities(self, board, valid_moves, temperature=1.0):
        """Get probability distribution over valid moves"""
        policy, value = self.predict(board)
        
        # Score each valid move
        move_scores = []
        for x, y, digit in valid_moves:
            idx = x * 81 + y * 9 + (digit - 1)
            move_scores.append(policy[idx])
        
        move_scores = np.array(move_scores)
        
        if temperature == 0:
            # Greedy
            best_idx = np.argmax(move_scores)
            probs = np.zeros(len(move_scores))
            probs[best_idx] = 1.0
        else:
            # Softmax with temperature
            move_scores = move_scores / temperature
            exp_scores = np.exp(move_scores - np.max(move_scores))
            probs = exp_scores / exp_scores.sum()
        
        return probs, value


class SelfPlayGenerator:
    """Generate training data through self-play"""
    
    def __init__(self, network):
        self.network = network
    
    def play_episode(self, puzzle, max_moves=50, exploration_temp=0.5):
        """Play one episode and collect training examples"""
        env = SudokuEnvironment(puzzle)
        examples = []
        
        for move_num in range(max_moves):
            if env.is_solved():
                # Success! Label all examples as positive
                for ex in examples:
                    ex.outcome = 1.0
                return examples, True
            
            if not env.is_valid():
                # Dead end - label as negative
                for ex in examples:
                    ex.outcome = -1.0
                return examples, False
            
            # Get strategic moves (cells with fewest candidates)
            valid_moves = env.get_cells_with_fewest_candidates()
            
            if not valid_moves:
                # No moves available
                progress = np.count_nonzero(env.board) / 81.0
                outcome = 2 * progress - 1
                for ex in examples:
                    ex.outcome = outcome
                return examples, False
            
            # Get move probabilities from network
            probs, value = self.network.get_move_probabilities(
                env.board, valid_moves, temperature=exploration_temp
            )
            
            # Store training example
            # We'll create a target distribution showing which moves were available
            target_policy = np.zeros(729)
            for prob, (x, y, digit) in zip(probs, valid_moves):
                idx = x * 81 + y * 9 + (digit - 1)
                target_policy[idx] = prob
            
            examples.append(TrainingExample(
                state=env.board.copy(),
                target_cell=None,
                target_digit=None,
                outcome=0.0  # Will be updated at end
            ))
            
            # Actually we need to store the policy, not individual cell/digit
            # Let me fix this by using a different structure
            examples[-1] = {
                'state': env.board.copy(),
                'policy': target_policy,
                'value': 0.0
            }
            
            # Sample move from probability distribution
            move_idx = np.random.choice(len(valid_moves), p=probs)
            x, y, digit = valid_moves[move_idx]
            
            # Make the move
            env.make_move(x, y, digit)
        
        # Didn't solve - partial credit
        progress = np.count_nonzero(env.board) / 81.0
        outcome = 2 * progress - 1
        
        for ex in examples:
            ex['value'] = outcome
        
        return examples, False
    
    def generate_training_data(self, num_episodes=100, difficulty=30):
        """Generate training data from multiple episodes"""
        from game import Backend
        
        all_examples = []
        successes = 0
        
        print(f"\n  Generating {num_episodes} episodes...")
        
        for ep in range(num_episodes):
            # Generate puzzle
            backend = Backend()
            backend.generate_sudoku(difficulty)
            puzzle = backend.get_state().board
            
            # Play episode
            examples, success = self.play_episode(
                puzzle, 
                max_moves=30,
                exploration_temp=0.8  # High temperature for exploration
            )
            
            if success:
                successes += 1
            
            all_examples.extend(examples)
            
            if (ep + 1) % 10 == 0:
                avg_examples = len(all_examples) / (ep + 1)
                print(f"    {ep+1}/{num_episodes} | "
                      f"Success: {successes}/{ep+1} | "
                      f"Avg examples/episode: {avg_examples:.1f}")
        
        print(f"  Total: {len(all_examples)} examples, {successes}/{num_episodes} solved")
        return all_examples


class SelfSupervisedTrainer:
    """Main training loop"""
    
    def __init__(self, model, buffer_size=50000):
        self.model = model
        self.network = PolicyValueNetwork(model)
        self.generator = SelfPlayGenerator(self.network)
        self.replay_buffer = deque(maxlen=buffer_size)
    
    def train_iteration(self, num_episodes=100, difficulty=30):
        """One training iteration"""
        # Generate data
        examples = self.generator.generate_training_data(num_episodes, difficulty)
        
        # Add to buffer
        for ex in examples:
            self.replay_buffer.append(ex)
        
        # Train network
        if len(self.replay_buffer) >= 128:
            self._train_network()
        
        return len(examples)
    
    def _train_network(self, batch_size=128, epochs=3):
        """Train network on replay buffer"""
        # Sample batch
        sample_size = min(batch_size * 5, len(self.replay_buffer))
        batch = random.sample(list(self.replay_buffer), sample_size)
        
        states = np.array([ex['state'].reshape(9, 9, 1) for ex in batch])
        policies = np.array([ex['policy'] for ex in batch])
        values = np.array([ex['value'] for ex in batch])
        
        print(f"  Training on {len(batch)} examples...")
        
        history = self.model.fit(
            states,
            [policies, values],
            batch_size=batch_size,
            epochs=epochs,
            verbose=0,
            validation_split=0.1
        )
        
        print(f"    Policy loss: {history.history['policy_loss'][-1]:.4f}")
        print(f"    Value loss: {history.history['value_loss'][-1]:.4f}")
        
        return history
    
    def train(self, iterations=20, episodes_per_iter=50, difficulty=30):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("SELF-SUPERVISED SUDOKU TRAINING")
        print(f"{'='*60}")
        print("Learning from scratch - no expert data!")
        print(f"Iterations: {iterations}")
        print(f"Episodes per iteration: {episodes_per_iter}")
        print(f"{'='*60}")
        
        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Generate and train
            num_examples = self.train_iteration(episodes_per_iter, difficulty)
            
            elapsed = time.time() - start_time
            print(f"\n  Iteration time: {elapsed:.1f}s")
            print(f"  Buffer size: {len(self.replay_buffer)}")
            
            # Evaluate every 5 iterations
            if (iteration + 1) % 5 == 0:
                print(f"\n  Evaluating...")
                self._evaluate(num_puzzles=20, difficulty=difficulty)
                self.model.save(f"sudoku_selfsup_iter_{iteration+1}.h5")
                print(f"  ✓ Checkpoint saved")
        
        print(f"\n{'='*60}")
        print("Training complete!")
        self.model.save("sudoku_selfsup_final.h5")
    
    def _evaluate(self, num_puzzles=20, difficulty=30):
        """Quick evaluation"""
        from game import Backend
        
        solved = 0
        total_moves = []
        
        for _ in range(num_puzzles):
            backend = Backend()
            backend.generate_sudoku(difficulty)
            puzzle = backend.get_state().board
            
            env = SudokuEnvironment(puzzle)
            
            for move_num in range(50):
                if env.is_solved():
                    solved += 1
                    total_moves.append(move_num)
                    break
                
                valid_moves = env.get_cells_with_fewest_candidates()
                if not valid_moves:
                    break
                
                probs, _ = self.network.get_move_probabilities(
                    env.board, valid_moves, temperature=0.0
                )
                
                best_move_idx = np.argmax(probs)
                x, y, digit = valid_moves[best_move_idx]
                env.make_move(x, y, digit)
                
                if not env.is_valid():
                    break
        
        accuracy = solved / num_puzzles * 100
        avg_moves = np.mean(total_moves) if total_moves else 0
        
        print(f"    Accuracy: {accuracy:.1f}% ({solved}/{num_puzzles})")
        if total_moves:
            print(f"    Avg moves: {avg_moves:.1f}")


def make_sudoku_model():
    """Create model for self-supervised learning"""
    input_board = layers.Input(shape=(9, 9, 1))

    x = layers.Conv2D(128, 3, padding="same")(input_board)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # Residual blocks
    for _ in range(8):
        skip = x
        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(128, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Add()([skip, x])
        x = layers.ReLU()(x)

    # Policy head - which move to make
    p = layers.Conv2D(32, 1, padding="same")(x)
    p = layers.BatchNormalization()(p)
    p = layers.ReLU()(p)
    p = layers.Flatten()(p)
    p = layers.Dense(729, activation="softmax", name="policy")(p)

    # Value head - how good is this position
    v = layers.Conv2D(32, 1, padding="same")(x)
    v = layers.BatchNormalization()(v)
    v = layers.ReLU()(v)
    v = layers.Flatten()(v)
    v = layers.Dense(128, activation="relu")(v)
    v = layers.Dropout(0.3)(v)
    v = layers.Dense(1, activation="tanh", name="value")(v)

    model = models.Model(inputs=input_board, outputs=[p, v])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss={'policy': 'categorical_crossentropy', 'value': 'mse'},
        loss_weights={'policy': 1.0, 'value': 0.5}
    )
    
    return model


if __name__ == "__main__":
    print("="*60)
    print("SELF-SUPERVISED SUDOKU LEARNING")
    print("="*60)
    print("\nNo expert data needed - learning from experience!")
    
    # Create model
    model = make_sudoku_model()
    print(f"\nModel parameters: {model.count_params():,}")
    
    # Train
    trainer = SelfSupervisedTrainer(model)
    trainer.train(
        iterations=25,
        episodes_per_iter=40,
        difficulty=32
    )
    
    print("\n" + "="*60)
    print("Training complete! Model learned from self-play.")
    print("="*60)