import chess
import numpy as np
import chess.svg 

import pickle


def generate_chessboard():
    class ChessMateAnalyzer:
        def __init__(self, matrix):
            self.matrix = matrix
            self.piece_map = {
                1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
                -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
                0: ''
            }
                
        def is_valid_position(self):

            if np.any(self.matrix[0] == 1) or np.any(self.matrix[7] == 1) or np.any(self.matrix[0] == -1) or np.any(self.matrix[7] == -1):
                return False

            white_kings = np.sum(self.matrix == 6)
            black_kings = np.sum(self.matrix == -6)
            white_knights = np.sum(self.matrix == 2)
            black_knights = np.sum(self.matrix == -2)
            white_rooks = np.sum(self.matrix == 4)
            black_rooks = np.sum(self.matrix == -4)
            white_queen = np.sum(self.matrix == 5)
            black_queen = np.sum(self.matrix == -5)

            if (white_knights > 2 or black_knights > 2 or 
                white_rooks > 2 or black_rooks > 2 or
                white_queen >1 or black_queen>1 or
                white_kings != 1 or black_kings != 1):
                return False

            white_light_squares_bishops = np.sum((self.matrix == 3) & ((np.arange(8) + np.arange(8)[:, None]) % 2 == 0))
            white_dark_squares_bishops = np.sum((self.matrix == 3) & ((np.arange(8) + np.arange(8)[:, None]) % 2 == 1))
            black_light_squares_bishops = np.sum((self.matrix == -3) & ((np.arange(8) + np.arange(8)[:, None]) % 2 == 0))
            black_dark_squares_bishops = np.sum((self.matrix == -3) & ((np.arange(8) + np.arange(8)[:, None]) % 2 == 1))

            if (white_light_squares_bishops > 1 or white_dark_squares_bishops > 1 or
                black_light_squares_bishops > 1 or black_dark_squares_bishops > 1):
                return False

            return True

        def is_checkmate(self):
            board = chess.Board(fen=self.matrix_to_fen())
            return board.is_checkmate()

        def classify_mate(self):
            king_pos = self.find_piece_positions(-6)

            for piece, name in [(5, 'queen_mate'), (4, 'rook_mate'), (3, 'bishop_mate'),
                                (2, 'knight_mate'), (1, 'pawn_mate')]:
                positions = self.find_piece_positions(piece)
                if positions is not None:
                    for pos in positions:
                        if self.is_attacking_king(piece, pos, king_pos):
                            return name,pos
            return None,None

        def is_attacking_king(self, piece, pos, king_pos):
            if piece == 5:  # Queen
                return self.is_adjacent(pos, king_pos , diagonal=True , straight=True) 
            elif piece == 4:  # Rook
                return self.is_adjacent(pos, king_pos, diagonal=False , straight=True) 
            elif piece == 3:  # Bishop
                return self.is_adjacent(pos, king_pos, diagonal=True, straight=False) 
            elif piece == 2:  # Knight
                return self.is_knight_attacking(pos, king_pos)
            elif piece == 1: # Pawn
                return self.is_pawn_attacking(pos, king_pos)
            return False

        def is_adjacent(self, pos1, pos2, diagonal, straight):
            matrix = self.matrix
            x1, y1 = pos1
            x2, y2 = pos2[0]

            def in_bounds(x, y):
                return 0 <= x < len(matrix) and 0 <= y < len(matrix[0])

            def is_straight_line_clear(num,p1, p2):
                x1, y1 = p1
                x2, y2 = p2

                if num==5:

                    if x1 == x2:  
                        if y1 > y2:
                            y1, y2 = y2, y1
                        return all(matrix[x1][y] == 0 for y in range(y1 + 1, y2))
                    elif y1 == y2:  
                        if x1 > x2:
                            x1, x2 = x2, x1
                        return all(matrix[x][y1] == 0 for x in range(x1 + 1, x2))
                    elif abs(x1 - x2) == abs(y1 - y2):
                        if x1 > x2:
                            x1, x2 = x2, x1
                            y1, y2 = y2, y1
                        if (x2 - x1) == (y2 - y1):
                            return all(matrix[x1 + i][y1 + i] == 0 for i in range(1, x2 - x1))
                        else:
                            return all(matrix[x1 + i][y1 - i] == 0 for i in range(1, x2 - x1))

                elif num==4:
                    if x1 == x2:  
                        if y1 > y2:
                            y1, y2 = y2, y1
                        return all(matrix[x1][y] == 0 for y in range(y1 + 1, y2))
                    elif y1 == y2:  
                        if x1 > x2:
                            x1, x2 = x2, x1
                        return all(matrix[x][y1] == 0 for x in range(x1 + 1, x2))

                elif num==3:
                    if abs(x1 - x2) == abs(y1 - y2):
                        if x1 > x2:
                            x1, x2 = x2, x1
                            y1, y2 = y2, y1
                        if (x2 - x1) == (y2 - y1):
                            return all(matrix[x1 + i][y1 + i] == 0 for i in range(1, x2 - x1))
                        else:
                            return all(matrix[x1 + i][y1 - i] == 0 for i in range(1, x2 - x1))
                        
                return False

            if diagonal and straight:
                if (abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1):
                    return True
                return is_straight_line_clear(5,pos1, pos2[0])

            if not diagonal and straight:
                if (x1 == x2 and abs(y1 - y2) == 1) or (y1 == y2 and abs(x1 - x2) == 1):
                    return True
                return is_straight_line_clear(4,pos1, pos2[0])

            if diagonal and not straight:
                if abs(x1 - x2) == 1 and abs(y1 - y2) == 1:
                    return True
                return is_straight_line_clear(3,pos1, pos2[0])

            return False

        def is_knight_attacking(self, knight_pos, king_pos):

            knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
            possible_attacks = [(knight_pos[0] + dx, knight_pos[1] + dy) for dx, dy in knight_moves]
            return king_pos[0] in possible_attacks

        def is_pawn_attacking(self, pawn_pos, king_pos, pawn_color='white'):

            attack_offsets = [(-1, 1), (-1, -1)] 
            possible_attacks = [(pawn_pos[0] + dx, pawn_pos[1] + dy) for dx, dy in attack_offsets]
            return king_pos[0] in possible_attacks

        def find_piece_positions(self, piece):
            pieces=[]
            for x in range(8):
                for y in range(8):
                    if self.matrix[x][y] == piece:

                        pieces.append((x,y))

            if len(pieces)>0:
                return pieces
            else:
                return None

        def is_white_mate(self):
            return any(6 in row for row in self.matrix)

        def matrix_to_fen(self):
            piece_to_char = {
                1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
                -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
                0: ''
            }

            fen_rows = []
            for row in self.matrix:
                fen_row = ''
                empty_count = 0
                for value in row:
                    piece = piece_to_char.get(value, '')
                    if piece:
                        if empty_count > 0:
                            fen_row += str(empty_count)
                            empty_count = 0
                        fen_row += piece
                    else:
                        empty_count += 1
                if empty_count > 0:
                    fen_row += str(empty_count)
                fen_rows.append(fen_row)

            fen = '/'.join(fen_rows)
            fen=fen+ " b - - 0 1"
            return fen

        def pre_mate(self):
            if not self.is_valid_position() or not self.is_checkmate():
                return None

            mate_type,pos = self.classify_mate()
            if not mate_type:
                return None

            return self.generate_pre_mate_position(mate_type,pos)

        def generate_pre_mate_position(self, mate_type, pos):
            king_pos = self.find_piece_positions(-6)
            piece_map = {
                'queen_mate': (5, self.move_queen),
                'rook_mate': (4, self.move_rook),
                'bishop_mate': (3, self.move_bishop),
                'knight_mate': (2, self.move_knight),
                'pawn_mate': (1, self.move_pawn)
            }
            piece, move_function = piece_map[mate_type]
            piece_pos = pos
            return move_function(piece_pos, king_pos)

        def move_queen(self, piece_pos, king_pos):
            return self.move_piece(piece_pos, king_pos, directions=[(1, 0), (0, 1), (1, 1), (1, -1),(-1,0),(0,-1),(-1,-1),(-1,1)])

        def move_rook(self, piece_pos, king_pos):
            return self.move_piece(piece_pos, king_pos, directions=[(1, 0), (0, 1),(-1,0),(0,-1)])

        def move_bishop(self, piece_pos, king_pos):
            return self.move_piece(piece_pos, king_pos, directions=[(1, 1), (1, -1),(-1,-1),(-1,1)])

        def move_knight(self, piece_pos, king_pos):
            knight_moves = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
            possible_positions = []

            for move in knight_moves:
                new_pos = (piece_pos[0] + move[0], piece_pos[1] + move[1])
                if self.is_within_bounds(new_pos) and self.matrix[new_pos[0]][new_pos[1]] == 0:
                    possible_positions.append(new_pos)

            if possible_positions:
                new_pos = possible_positions[np.random.randint(len(possible_positions))]
                return self.create_new_board(piece_pos, new_pos, piece=2)

            return None

        def move_pawn(self, piece_pos, king_pos):

            direction = -1 if self.matrix[piece_pos[0]][piece_pos[1]] > 0 else 1
            new_pos = (piece_pos[0] - direction, piece_pos[1])

            if self.is_within_bounds(new_pos) and self.matrix[new_pos[0]][new_pos[1]] == 0:
                return self.create_new_board(piece_pos, new_pos, piece=self.matrix[piece_pos[0]][piece_pos[1]])

            diagonal_left = (piece_pos[0] - direction, piece_pos[1] - 1)
            diagonal_right = (piece_pos[0] - direction, piece_pos[1] + 1)

            if self.is_within_bounds(diagonal_left) and self.matrix[diagonal_left[0]][diagonal_left[1]] == 0:
                return self.create_new_board(piece_pos, diagonal_left, piece=self.matrix[piece_pos[0]][piece_pos[1]],p_dir=1)

            if self.is_within_bounds(diagonal_right) and self.matrix[diagonal_right[0]][diagonal_right[1]] == 0:
                return self.create_new_board(piece_pos, diagonal_right, piece=self.matrix[piece_pos[0]][piece_pos[1]],p_dir=1)

            return None


        def move_piece(self, piece_pos, king_pos, directions):
            possible_positions = []

            for direction in directions:
                new_pos = (piece_pos[0], piece_pos[1])

                while True:
                    next_pos = (new_pos[0] + direction[0], new_pos[1] + direction[1])

                    if not self.is_within_bounds(next_pos):
                        break

                    if self.matrix[next_pos[0]][next_pos[1]] == 0:
                        possible_positions.append(next_pos)
                        new_pos = next_pos
                    else:
                        break

            if possible_positions:
                new_pos = possible_positions[np.random.randint(len(possible_positions))]
                return self.create_new_board(piece_pos, new_pos, piece=self.matrix[piece_pos[0]][piece_pos[1]])

            return None

        def is_within_bounds(self, pos):
            return 0 <= pos[0] < 8 and 0 <= pos[1] < 8

        def create_new_board(self, piece_pos, new_pos, piece,p_dir=0):

            def matrix_to_fen_local(matrix):
                piece_to_char = {
                    1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
                    -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
                    0: ''
                }

                fen_rows = []
                for row in matrix:
                    fen_row = ''
                    empty_count = 0
                    for value in row:
                        piece = piece_to_char.get(value, '')
                        if piece:
                            if empty_count > 0:
                                fen_row += str(empty_count)
                                empty_count = 0
                            fen_row += piece
                        else:
                            empty_count += 1
                    if empty_count > 0:
                        fen_row += str(empty_count)
                    fen_rows.append(fen_row)

                fen = '/'.join(fen_rows)
                fen=fen+ " b - - 0 1"
                return fen

            def is_in_check(matrix):
                board = chess.Board(fen=matrix_to_fen_local(matrix))
                return board.is_check()

            new_board = [row[:] for row in self.matrix]
            new_board[piece_pos[0]][piece_pos[1]]=0
            new_board[new_pos[0]][new_pos[1]] = piece

            if is_in_check(new_board):

                if piece != 1:
                    new_board = [row[:] for row in self.matrix]
                    if piece_pos[0]==0 or piece_pos[0]==7:
                        new_board[piece_pos[0]][piece_pos[1]] = np.random.choice([-2,-3,-4])
                    else:
                        new_board[piece_pos[0]][piece_pos[1]] = np.random.choice([-1,-2,-3,-4])
                    new_board[new_pos[0]][new_pos[1]] = piece
                    self.matrix=new_board
                    return self.matrix_to_fen()
                else:

                    new_board = [row[:] for row in self.matrix]
                    if p_dir==1:
                        new_board[piece_pos[0]][piece_pos[1]] = np.random.choice([-1,-2,-3,-4])
                    else:
                        new_board[piece_pos[0]][piece_pos[1]] = 0

                    new_board[new_pos[0]][new_pos[1]] = piece
                    self.matrix=new_board
                    return self.matrix_to_fen()

            else:
                if piece != 1:
                    new_board = [row[:] for row in self.matrix]
                    if piece_pos[0]==0 or piece_pos[0]==7:
                        new_board[piece_pos[0]][piece_pos[1]] = 0
                    else:
                        new_board[piece_pos[0]][piece_pos[1]] = 0
                    new_board[new_pos[0]][new_pos[1]] = piece
                    self.matrix=new_board
                    return self.matrix_to_fen()
                else:

                    new_board = [row[:] for row in self.matrix]
                    if p_dir==1:
                        new_board[piece_pos[0]][piece_pos[1]] = np.random.choice([-1,-2,-3,-4])
                    else:
                        new_board[piece_pos[0]][piece_pos[1]] = 0

                    new_board[new_pos[0]][new_pos[1]] = piece
                    self.matrix=new_board
                    return self.matrix_to_fen()


    def generate_matrix(generator):
        valid_values = np.array([0, 1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6])
        noise = np.random.normal(0, 1, (1, 100))
        generated_logits = generator.predict(noise,verbose=0)
        generated_matrix = np.argmax(generated_logits, axis=-1)
        return np.vectorize(lambda x: valid_values[x])(generated_matrix[0])

    def is_check_irrespective_of_side(fen):
        board = chess.Board(fen)
        side_to_move = board.turn
        is_check_side_to_move = board.is_check()
        board.turn = not side_to_move
        is_check_opposite_side = board.is_check()
        board.turn = side_to_move
        return is_check_side_to_move or is_check_opposite_side

    def is_white_king_in_check(board):
        # Get the square of the white king
        white_king_square = board.king(chess.WHITE)

        # Check if the white king is in check
        if board.is_check():
            return board.is_attacked_by(chess.BLACK, white_king_square)

        return False

    def matrix_to_fen_local2(matrix):
        piece_to_char = {
            1: 'P', 2: 'N', 3: 'B', 4: 'R', 5: 'Q', 6: 'K',
            -1: 'p', -2: 'n', -3: 'b', -4: 'r', -5: 'q', -6: 'k',
            0: ''
        }

        fen_rows = []
        for row in matrix:
            fen_row = ''
            empty_count = 0
            for value in row:
                piece = piece_to_char.get(value, '')
                if piece:
                    if empty_count > 0:
                        fen_row += str(empty_count)
                        empty_count = 0
                    fen_row += piece
                else:
                    empty_count += 1
            if empty_count > 0:
                fen_row += str(empty_count)
            fen_rows.append(fen_row)

        fen = '/'.join(fen_rows)
        fen=fen+ " b - - 0 1"
        return fen

    
    
    def load_generator():
        paths = [
            r"artifacts\bishop_mate_generator.pkl",
            r"artifacts\knight_mate_generator.pkl",
            r"artifacts\pawn_mate_generator.pkl",
            r"artifacts\rook_mate_generator.pkl",
            r"artifacts\queen_mate_generator.pkl"
        ]
        path = np.random.choice(paths)
        with open(path, 'rb') as file:
            return pickle.load(file)
    while True:
        generator = load_generator()
        matrix = generate_matrix(generator)
        initial_matrix=np.copy(matrix)
        analyzer = ChessMateAnalyzer(matrix)
        pre_mate_position = analyzer.pre_mate()
        fen=matrix_to_fen_local2(initial_matrix)
        board = chess.Board(fen)

        if pre_mate_position is not None and not is_check_irrespective_of_side(pre_mate_position) and not is_white_king_in_check(board):
            return(pre_mate_position)
        
if __name__ == "__main__":
    fen = generate_chessboard()
    print(fen)