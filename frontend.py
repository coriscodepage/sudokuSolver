import sys

import keras
from game import Backend, Coordinates, SetSquareCommand
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel, QPushButton, QVBoxLayout, QHBoxLayout, QLineEdit
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtCore import Qt
from typing import Optional

from inference import Infer

class SudokuCell(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(50, 50)
        font = QFont("Arial", 20)
        self.setFont(font)
        self.setAlignment(Qt.AlignCenter) # type: ignore
        self.border_style = ""

    def set_value(self, value: str, is_given=False, is_correct=True):
        if value != "0":
            self.setText(str(value))
            font = QFont("Arial", 20)
            if is_given:
                font = QFont("Arial", 22)
                font.setBold(True)
            else:
                font.setBold(False)
            self.setFont(font)
            color = "" if is_correct else "red"
            self.setStyleSheet(self.border_style + f"color: {color};")
        else:
            self.setText("")

class SudokuBoard(QWidget):
    def __init__(self, backend, infer):
        super().__init__()
        self.backend: Backend = backend
        self.infer: Infer = infer
        self.cells: list[list[Optional[SudokuCell]]] = [[None for _ in range(9)] for _ in range(9)]
        self.history: list[SetSquareCommand] = []
        self.index: int = 0
        self.solved: bool = False

        main_layout = QVBoxLayout()
        
        grid = QGridLayout()
        grid.setSpacing(0)

        for row in range(9):
            for col in range(9):
                cell = SudokuCell()
                self.cells[row][col] = cell
                style = "border: 1px solid;"
                if row % 3 == 0:
                    style += "border-top: 2px solid;"
                if col % 3 == 0:
                    style += "border-left: 2px solid;"
                if row == 8:
                    style += "border-bottom: 2px solid;"
                if col == 8:
                    style += "border-right: 2px solid;"

                cell.border_style = style
                cell.setStyleSheet(style)

                grid.addWidget(cell, row, col)

        main_layout.addLayout(grid)

        self.status_label = QLabel("Sudoku not solved")
        font = QFont("Arial", 12)
        font.setBold(True)
        self.status_label.setFont(font)
        main_layout.addWidget(self.status_label)

        difficulty_layout = QHBoxLayout()
        difficulty_label = QLabel("Difficulty (0-81):")
        self.difficulty_input = QLineEdit()
        self.difficulty_input.setValidator(QIntValidator(0, 81))
        self.difficulty_input.textChanged.connect(self.enforce_difficulty_limit)
        self.difficulty_input.setText("40")
        self.difficulty_input.setMaximumWidth(80)
        difficulty_layout.addWidget(difficulty_label)
        difficulty_layout.addWidget(self.difficulty_input)
        difficulty_layout.addStretch()
        main_layout.addLayout(difficulty_layout)


        controls_layout = QHBoxLayout()
        
        self.btn_back = QPushButton("Back")
        self.btn_back.clicked.connect(self.on_back)
        controls_layout.addWidget(self.btn_back)

        self.btn_solve = QPushButton("Solve")
        self.btn_solve.clicked.connect(self.on_solve)
        controls_layout.addWidget(self.btn_solve)

        self.btn_forward = QPushButton("Forward")
        self.btn_forward.clicked.connect(self.on_forward)
        controls_layout.addWidget(self.btn_forward)

        self.btn_generate = QPushButton("Generate New")
        self.btn_generate.clicked.connect(self.on_generate)
        controls_layout.addWidget(self.btn_generate)

        main_layout.addLayout(controls_layout)

        self.setLayout(main_layout)
        self.setWindowTitle("Sudoku")
        self.setFixedSize(460, 510)

    def use_model(self):
        if not self.solved:
            self.status_label.setText("Generating solution...")
            QApplication.processEvents()
            result, self.history = self.infer.generate_steps(self.backend.get_state())
            self.status_label.setText("Correct?: " + ("yes" if result.check_correct() else "no"))
            self.solved = True

    def on_solve(self):
        self.use_model()
        while self.index < len(self.history):
            self.display_step()

    def on_back(self):
        if self.index > 0:
            self.index -= 1
            command = self.history[self.index]
            cell = self.cells[command.coords.x][command.coords.y]
            if cell is not None:
                cell.set_value("0")           

    def on_forward(self):
        if self.history == []:
            self.use_model()
        self.display_step()

    def on_generate(self):
        self.history = []
        self.index = 0
        self.solved = False
        self.status_label.setText("Sudoku not solved")
        difficulty = int(self.difficulty_input.text())
        self.backend.generate_sudoku(difficulty)
        self.display_board()        

    def display_step(self):
        if self.index < len(self.history):
            command = self.history[self.index]
            self.index += 1
            cell = self.cells[command.coords.x][command.coords.y]
            if cell is not None:
                cell.set_value(str(command.digit), False, command.valid)

    def display_board(self):
        for row in range(9):
            for col in range(9):
                token = self.backend.get_state().board[row][col]
                cell = self.cells[row][col]
                if cell is not None:
                    cell.set_value(str(token), True)

    def enforce_difficulty_limit(self):
        text = self.difficulty_input.text()
        if text and text.isdigit():
            value = int(text)
            if value > 81:
                self.difficulty_input.setText('81')
                self.difficulty_input.setCursorPosition(len('81'))

if __name__ == "__main__":
    MODEL_PATH = "best_weights.keras"
    app = QApplication(sys.argv)
    backend = Backend()
    model = keras.models.load_model(MODEL_PATH)
    infer = Infer(model) # type: ignore
    window = SudokuBoard(backend, infer)
    window.show()
    backend.generate_sudoku()
    window.display_board()
    sys.exit(app.exec_())