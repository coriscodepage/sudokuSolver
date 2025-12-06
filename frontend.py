import sys
from game import Backend, Coordinates, SetSquareCommand
from PySide6.QtWidgets import QApplication, QWidget, QGridLayout, QLabel
from PySide6.QtGui import QFont, QIntValidator
from PySide6.QtCore import Qt

class SudokuCell(QLabel):
    def __init__(self):
        super().__init__()
        self.setFixedSize(50, 50)
        self.setFont(QFont("Arial", 20))
        self.setAlignment(Qt.AlignCenter)



class SudokuBoard(QWidget):
    def __init__(self, backend):
        super().__init__()
        self.backend: Backend = backend

        self.cells = [[None]*9 for _ in range(9)]
        grid = QGridLayout()
        grid.setSpacing(0)

        for row in range(9):
            for col in range(9):
                cell = SudokuCell()
                self.cells[row][col] = cell
                style = "border: 1px solid black;"
                if row % 3 == 0:
                    style += "border-top: 2px solid black;"
                if col % 3 == 0:
                    style += "border-left: 2px solid black;"
                if row == 8:
                    style += "border-bottom: 2px solid black;"
                if col == 8:
                    style += "border-right: 2px solid black;"

                cell.setStyleSheet(style)

                grid.addWidget(cell, row, col)

        self.setLayout(grid)
        self.setWindowTitle("Sudoku")
        self.setFixedSize(460, 460)

    def display_board(self):
        for row in range(9):
            for col in range(9):
                token = self.backend.get_state().board[row][col]
                if token != 0:
                    self.cells[row][col].setText(str(token))
                else:
                    self.cells[row][col].setText("")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    backend = Backend()
    window = SudokuBoard(backend)
    window.show()
    window.display_board()
    #cmd = SetSquareCommand(backend, Coordinates(2, 0), Token(5))
    #result = cmd.execute()
    backend.generate_sudoku()
    window.display_board()
    sys.exit(app.exec_())