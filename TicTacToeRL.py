import numpy as np
import pandas as pd
import random
import json
from typing import List, Tuple

class TicTacToe:
    def __init__(self):
        self.board = np.full(9, ' ')
        self.current_winner = None

    def available_moves(self) -> np.ndarray:
        return np.where(self.board == ' ')[0]

    def make_move(self, square: int, letter: str) -> bool:
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(square, letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, square: int, letter: str) -> bool:
        row_ind = square // 3
        col_ind = square % 3

        # Check row
        if np.all(self.board[row_ind * 3:(row_ind + 1) * 3] == letter):
            return True

        # Check column
        if np.all(self.board[col_ind::3] == letter):
            return True

        # Check diagonals
        if square % 2 == 0:
            if np.all(self.board[[0, 4, 8]] == letter) or np.all(self.board[[2, 4, 6]] == letter):
                return True

        return False

    def is_full(self) -> bool:
        return np.all(self.board != ' ')

    def reset(self):
        self.board = np.full(9, ' ')
        self.current_winner = None

    def display_board(self):
        print("\n".join([" | ".join(self.board[i * 3:(i + 1) * 3]) for i in range(3)]))
        print("-" * 9)

    def get_state(self) -> str:
        return ''.join(self.board)

class QLearningAgent:
    def __init__(self, epsilon=0.1, alpha=0.5, gamma=0.9):
        self.q_table = pd.DataFrame(columns=["state", "action", "value"]).set_index(["state", "action"])
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

    def choose_action(self, state: str, available_moves: List[int]) -> int:
        if random.random() < self.epsilon:
            return random.choice(available_moves)

        q_values = self.q_table.loc[(state,), :].reindex(available_moves, fill_value=0)
        max_q = q_values["value"].max()
        best_actions = q_values[q_values["value"] == max_q].index.get_level_values("action")
        return random.choice(best_actions)

    def update_q_value(self, state: str, action: int, reward: float, next_state: str, next_available_moves: List[int]):
        max_future_q = (
            self.q_table.loc[(next_state,), "value"]
            .reindex(next_available_moves, fill_value=0)
            .max()
        )
        current_q = self.q_table.loc[(state, action), "value"] if (state, action) in self.q_table.index else 0
        new_q = current_q + self.alpha * (reward + self.gamma * max_future_q - current_q)
        self.q_table.loc[(state, action), "value"] = new_q

    def save_q_table(self, filename: str):
        self.q_table.to_csv(filename)

    def load_q_table(self, filename: str):
        try:
            self.q_table = pd.read_csv(filename).set_index(["state", "action"])
        except FileNotFoundError:
            self.q_table = pd.DataFrame(columns=["state", "action", "value"]).set_index(["state", "action"])

def train_agent(agent: QLearningAgent, episodes: int, save_file: str):
    env = TicTacToe()

    for episode in range(episodes):
        env.reset()
        state = env.get_state()

        while True:
            available_moves = env.available_moves()
            action = agent.choose_action(state, available_moves)
            env.make_move(action, 'X')
            next_state = env.get_state()

            if env.current_winner == 'X':
                reward = 1
                agent.update_q_value(state, action, reward, next_state, [])
                break
            elif env.is_full():
                reward = 0.5
                agent.update_q_value(state, action, reward, next_state, [])
                break

            opp_action = random.choice(env.available_moves())
            env.make_move(opp_action, 'O')
            next_state = env.get_state()

            if env.current_winner == 'O':
                reward = -1
            else:
                reward = 0

            agent.update_q_value(state, action, reward, next_state, env.available_moves())
            state = next_state

        if episode % 100 == 0:
            agent.epsilon = max(0.01, agent.epsilon * 0.99)
            print(f"Episode {episode}/{episodes} complete. Epsilon: {agent.epsilon:.4f}")

    agent.save_q_table(save_file)

def play_against_agent(agent: QLearningAgent):
    env = TicTacToe()

    while True:
        env.display_board()
        available_moves = env.available_moves()
        move = int(input(f"Your move (0-8): "))
        while move not in available_moves:
            move = int(input("Invalid move. Try again: "))
        env.make_move(move, 'O')
        env.display_board()

        if env.current_winner == 'O':
            print("You win!")
            break
        elif env.is_full():
            print("It's a draw!")
            break

        state = env.get_state()
        agent_move = agent.choose_action(state, env.available_moves())
        env.make_move(agent_move, 'X')
        print("Agent's move:")
        env.display_board()

        if env.current_winner == 'X':
            print("Agent wins!")
            break
        elif env.is_full():
            print("It's a draw!")
            break

if __name__ == "__main__":
    q_table_file = "q_table.csv"
    agent = QLearningAgent()
    agent.load_q_table(q_table_file)

    while True:
        print("\n1. Train the agent")
        print("2. Play against the agent")
        print("3. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            episodes = int(input("Enter the number of training episodes: "))
            train_agent(agent, episodes, q_table_file)
        elif choice == "2":
            play_against_agent(agent)
        elif choice == "3":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
