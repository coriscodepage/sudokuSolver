import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from collections import deque
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class TrainingExample:
    """Stores one training example (state, policy, value)"""
    state: np.ndarray  # 9x9x1 board representation
    policy: np.ndarray  # 729-dim policy vector (9x9x9)
    value: float  # -1, 0, or 1

class NeuralMCTS:
    """MCTS enhanced with neural network for AlphaZero"""
    
    def __init__(self, state, model, parent=None, action=None, prior=0.0):
        self.state = state
        self.model = model
        self.parent = parent
        self.action = action
        self.prior = prior
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.untried_actions = None
        
    def expand(self):
        """Expand node using neural network predictions"""
        if self.untried_actions is None:
            # Get policy and value from neural network
            board_input = self.state.board.reshape(1, 9, 9, 1)
            policy_logits, value = self.model.predict(board_input, verbose=0)
            
            # Get available actions
            actions = self.state.available_actions()
            
            # Create children with network priors
            action_probs = self._get_action_probs(policy_logits[0], actions)
            
            for action, prob in zip(actions, action_probs):
                # Create new state
                new_state = self._apply_action(self.state, action)
                if new_state is not None:
                    child = NeuralMCTS(new_state, self.model, self, action, prob)
                    self.children.append(child)
            
            self.untried_actions = []
        
        return len(self.children) > 0
    
    def _get_action_probs(self, policy, actions):
        """Convert neural network policy to action probabilities"""
        # Policy is 729-dim: 9x9 grid, 9 possible digits
        probs = []
        for move in actions:
            idx = move.coords.x * 81 + move.coords.y * 9 + (move.digit - 1)
            probs.append(policy[idx])
        
        # Normalize
        probs = np.array(probs)
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(probs)) / len(probs)
        
        return probs
    
    def _apply_action(self, state, action):
        """Apply action and propagate constraints"""
        from game import GameState
        new_state = GameState()
        new_state.board = np.copy(state.board)
        new_state.board[action.coords.x][action.coords.y] = action.digit
        
        # Use propagation from your MCTSNode
        from mcts import MCTSNode
        if MCTSNode.propagate(new_state) is None:
            return None
        return new_state
    
    def select_child(self, c_puct=1.0):
        """Select child using PUCT algorithm (AlphaZero's UCB variant)"""
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_visits = np.sqrt(self.visits)
        
        for child in self.children:
            if child.visits == 0:
                q_value = 0
            else:
                q_value = child.value_sum / child.visits
            
            # PUCT formula: Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
            u_value = c_puct * child.prior * sqrt_parent_visits / (1 + child.visits)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def search(self, num_simulations=100):
        """Run MCTS simulations"""
        for _ in range(num_simulations):
            node = self
            
            # Selection: traverse to leaf
            while node.children and not node.state.is_finished():
                node = node.select_child()
            
            # Expansion
            if not node.state.is_finished():
                node.expand()
                if node.children:
                    node = node.children[0]
            
            # Evaluation (using neural network)
            if node.state.check_solved():
                value = 1.0
            else:
                board_input = node.state.board.reshape(1, 9, 9, 1)
                _, value_pred = node.model.predict(board_input, verbose=0)
                value = value_pred[0][0]
            
            # Backpropagation
            while node is not None:
                node.visits += 1
                node.value_sum += value
                node = node.parent
    
    def get_policy(self, temperature=1.0):
        """Get improved policy after search"""
        if not self.children:
            return None, None
        
        visits = np.array([child.visits for child in self.children])
        
        if temperature == 0:
            # Greedy selection
            best_idx = np.argmax(visits)
            probs = np.zeros(len(visits))
            probs[best_idx] = 1.0
        else:
            # Temperature-based selection
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
        
        return probs, self.children


class AlphaZeroTrainer:
    """Trainer for AlphaZero on Sudoku"""
    
    def __init__(self, model, buffer_size=10000):
        self.model = model
        self.replay_buffer = deque(maxlen=buffer_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
    def self_play(self, num_games=100, simulations_per_move=100):
        """Generate training data through self-play"""
        from game import Backend, GameState
        
        examples = []
        
        for game_num in range(num_games):
            backend = Backend()
            backend.generate_sudoku(65)  # Adjust difficulty
            game_state = backend.get_state()
            
            game_examples = []
            
            while not game_state.is_finished():
                # Run MCTS
                root = NeuralMCTS(game_state, self.model)
                root.search(simulations_per_move)
                
                # Get improved policy
                probs, children = root.get_policy(temperature=1.0)
                
                if probs is None:
                    break
                
                # Create policy vector (729-dim)
                policy_vector = np.zeros(729)
                for prob, child in zip(probs, children):
                    if child.action:
                        idx = (child.action.coords.x * 81 + 
                               child.action.coords.y * 9 + 
                               (child.action.digit - 1))
                        policy_vector[idx] = prob
                
                # Store training example
                game_examples.append(TrainingExample(
                    state=game_state.board.copy(),
                    policy=policy_vector,
                    value=0.0  # Will be updated at end
                ))
                
                # Select action
                action_idx = np.random.choice(len(probs), p=probs)
                selected_child = children[action_idx]
                game_state = selected_child.state
            
            # Update values based on game outcome
            if game_state.check_solved():
                value = 1.0
            else:
                value = -1.0
            
            for example in game_examples:
                example.value = value
                self.replay_buffer.append(example)
            
            examples.extend(game_examples)
            
            if (game_num + 1) % 10 == 0:
                print(f"Self-play game {game_num + 1}/{num_games}, "
                      f"Solved: {value > 0}, Examples: {len(game_examples)}")
        
        return examples
    
    def train_network(self, batch_size=32, epochs=10):
        """Train neural network on replay buffer"""
        if len(self.replay_buffer) < batch_size:
            return
        
        # Sample batch
        batch = random.sample(self.replay_buffer, min(batch_size, len(self.replay_buffer)))
        
        states = np.array([ex.state.reshape(9, 9, 1) for ex in batch])
        policies = np.array([ex.policy for ex in batch])
        values = np.array([ex.value for ex in batch])
        
        # Train
        history = self.model.fit(
            states,
            [policies, values],
            batch_size=batch_size,
            epochs=epochs,
            verbose=1
        )
        
        return history
    
    def train(self, iterations=100, games_per_iter=50, simulations=100):
        """Full training loop"""
        for iteration in range(iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{iterations}")
            print(f"{'='*60}")
            
            # Self-play
            print("Running self-play...")
            examples = self.self_play(games_per_iter, simulations)
            
            # Train network
            print(f"\nTraining network on {len(self.replay_buffer)} examples...")
            self.train_network(batch_size=64, epochs=5)
            
            # Save checkpoint
            if (iteration + 1) % 10 == 0:
                self.model.save(f"sudoku_alphazero_iter_{iteration + 1}.h5")
                print(f"Model saved at iteration {iteration + 1}")


def make_sudoku_alphazero_model():
    """Create AlphaZero model architecture"""
    input_board = layers.Input(shape=(9, 9, 1))

    x = layers.Conv2D(64, 3, padding="same", activation="relu")(input_board)
    x = layers.BatchNormalization()(x)

    # Residual Tower
    for _ in range(6):
        skip_connection = x
        
        h = layers.Conv2D(64, 3, padding="same")(x)
        h = layers.BatchNormalization()(h)
        h = layers.ReLU()(h)
        
        h = layers.Conv2D(64, 3, padding="same")(h)
        h = layers.BatchNormalization()(h)
        
        x = layers.Add()([skip_connection, h]) 
        x = layers.ReLU()(x)

    # Policy head
    p = layers.Conv2D(2, 1, padding="same", activation="relu")(x)
    p = layers.Flatten()(p)
    p = layers.Dense(729, activation="softmax", name="policy")(p)

    # Value head
    v = layers.Conv2D(1, 1, padding="same", activation="relu")(x)
    v = layers.Flatten()(v)
    v = layers.Dense(64, activation="relu")(v)
    v = layers.Dense(1, activation="tanh", name="value")(v)

    model = models.Model(inputs=input_board, outputs=[p, v])
    
    # Compile with custom losses
    model.compile(
        optimizer='adam',
        loss={
            'policy': 'categorical_crossentropy',
            'value': 'mse'
        },
        loss_weights={
            'policy': 1.0,
            'value': 1.0
        }
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Create model
    model = make_sudoku_alphazero_model()
    
    # Create trainer
    trainer = AlphaZeroTrainer(model, buffer_size=10000)
    
    # Train
    trainer.train(
        iterations=50,
        games_per_iter=20,
        simulations=80
    )
    
    # Save final model
    model.save("sudoku_alphazero_final.h5")