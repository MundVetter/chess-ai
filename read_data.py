import chess
import chess.pgn
import numpy as np

def pgn_to_bitboards(pgn_file):
    pgn = open(pgn_file)
    test = []
    train = []

    counter = 0
    while True:
        game = chess.pgn.read_game(pgn)
        counter += 1

        if game is None:
            break  # end of file

        if counter % 100 == 0:
            print("Processing game %d" % counter)

        board = game.board()
        movecounter = 0
        for move in game.mainline_moves():
            board.push(move)

            movecounter += 1
            if movecounter < 20:
                continue

            board_state_copy = np.zeros(6*64 + 6, dtype=float)
            for j in range(6):
                    board_state_copy[j*64:(j+1)*64] += bitboard_to_array(int(board.pieces(chess.Piece.from_symbol('PNBRQK'[j]).piece_type, chess.WHITE)))
                    board_state_copy[j*64:(j+1)*64] -= bitboard_to_array(int(board.pieces(chess.Piece.from_symbol('PNBRQK'[j]).piece_type, chess.BLACK)))

            board_state_copy[6*64] = float(board.turn)
            board_state_copy[6*64 + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
            board_state_copy[6*64 + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
            board_state_copy[6*64 + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
            board_state_copy[6*64 + 4] = float(board.has_queenside_castling_rights(chess.BLACK))

            # add the final result
            result = game.headers["Result"]
            if result == "1-0":
                board_state_copy[6*64 + 5] = 1
            elif result == "0-1":
                board_state_copy[6*64 + 5] = -1
            else:
                board_state_copy[6*64 + 5] = 0

            if counter < 1500:
                test.append(board_state_copy)
            else:
                train.append(board_state_copy)

        if counter > 100_000:
            break

    pgn.close()

    return np.array(test), np.array(train)

def bitboard_to_array(bb: int) -> np.ndarray:
    s = 8 * np.arange(7, -1, -1, dtype=np.uint64)
    b = (bb >> s).astype(np.uint8)
    b = np.unpackbits(b, bitorder="little")
    return b.reshape(64)

pgn_file = "data/lichess_db_standard_rated_2016-03.pgn"
test, train = pgn_to_bitboards(pgn_file)
# save numpy file
np.save("data/test2.npy", test)
np.save("data/train.npy", train)