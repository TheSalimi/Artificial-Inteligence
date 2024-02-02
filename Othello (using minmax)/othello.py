import copy

class Othello:
    def __init__(self):
        self.board = [[' ' for _ in range(8)] for _ in range(8)]
        self.board[3][3] = 'B'
        self.board[3][4] = 'W'
        self.board[4][3] = 'W'
        self.board[4][4] = 'B'
        self.current_player = 'W'

    def display_board(self):
        print("   0   1   2   3   4   5   6   7")
        print(" +---+---+---+---+---+---+---+---+")
        for i, row in enumerate(self.board):
            print(f"{i}|", end="")
            for cell in row:
                print(f" {cell} |", end="")
            print("\n +---+---+---+---+---+---+---+---+")
        print()

    def is_valid_move(self, row, col):
        if self.board[row][col] != ' ':
            return False
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.get_opponent():
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.get_opponent():
                    r, c = r + dr, c + dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player:
                    return True
        return False

    def get_opponent(self):
        return 'B' if self.current_player == 'W' else 'W'

    def make_move(self, row, col):
        if not self.is_valid_move(row, col):
            return False

        self.board[row][col] = self.current_player
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.get_opponent():
                to_flip = []
                while 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.get_opponent():
                    to_flip.append((r, c))
                    r, c = r + dr, c + dc
                if 0 <= r < 8 and 0 <= c < 8 and self.board[r][c] == self.current_player:
                    for flip_row, flip_col in to_flip:
                        self.board[flip_row][flip_col] = self.current_player

        self.current_player = self.get_opponent()
        return True

    def is_game_over(self):
        return all(' ' not in row for row in self.board) or not any(
            any(self.is_valid_move(r, c) for c in range(8)) for r in range(8)
        )

    def count_score(self):
        w_count = sum(row.count('W') for row in self.board)
        b_count = sum(row.count('B') for row in self.board)
        return w_count, b_count

def heuristic_evaluation(board):
    w_count, b_count = board.count_score()
    mobility_factor = len([(r, c) for r in range(8) for c in range(8) if board.is_valid_move(r, c)])
    corner_control_factor = 25 * (board.board[0][0] == 'B') + 25 * (board.board[0][7] == 'B') + \
                            25 * (board.board[7][0] == 'B') + 25 * (board.board[7][7] == 'B')
    return w_count - b_count + mobility_factor + corner_control_factor

def minmax(board, depth, maximizing_player, alpha, beta):
    if depth == 0 or board.is_game_over():
        return heuristic_evaluation(board)

    legal_moves = [(r, c) for r in range(8) for c in range(8) if board.is_valid_move(r, c)]

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            new_board = copy.deepcopy(board)
            new_board.make_move(*move)
            eval = minmax(new_board, depth - 1, False, alpha, beta)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            new_board = copy.deepcopy(board)
            new_board.make_move(*move)
            eval = minmax(new_board, depth - 1, True, alpha, beta)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def get_best_move(board, depth):
    legal_moves = [(r, c) for r in range(8) for c in range(8) if board.is_valid_move(r, c)]
    best_move = None
    best_eval = float('-inf')

    for move in legal_moves:
        new_board = copy.deepcopy(board)
        new_board.make_move(*move)
        eval = minmax(new_board, depth - 1, False, float('-inf'), float('inf'))

        if eval > best_eval:
            best_eval = eval
            best_move = move

    return best_move

def main():
    game = Othello()

    while not game.is_game_over():
        game.display_board()

        if game.current_player == 'W':
            row, col = map(int, input("Enter your move (row and column, separated by space): ").split())
            if not game.make_move(row, col):
                print("Invalid move. Try again.")
                continue
        else:
            print("AI is thinking...")
            row, col = get_best_move(game, depth=6)
            game.make_move(row, col)
            print(f"AI plays: {row}, {col}")

    game.display_board()
    w_count, b_count = game.count_score()
    print(f"Game over. Final score - W: {w_count}, B: {b_count}")
    if w_count > b_count:
        print("You win!")
    elif w_count < b_count:
        print("AI wins!")
    else:
        print("It's a tie!")

if __name__ == "__main__":
    main()
