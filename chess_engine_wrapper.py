from pystockfish import *
SEARCH_DEPTH = 19
SEARCH_TIME = 1000


class chess_engine_wrapper:

    def __init__(self):
         self.chess_engine = Engine(depth=SEARCH_DEPTH)
         self.moves = []

    def get_best_move(self, last_rival_move = None):
        if last_rival_move is not None:
            self.moves.append(last_rival_move)
            self.chess_engine.setposition(self.moves)
        best_move = (self.chess_engine.bestmove())['move']
        self.moves.append(best_move)
        return best_move

def chess_engine_tester():
    chess_engine = chess_engine_wrapper()
    while True:
        rival_move = input('insert_rival_move')
        best_move = chess_engine.get_best_move(rival_move)
        print(best_move)

# for carmel with luv
# chess_engine_tester()
# chess_engine = Engine(depth=SEARCH_DEPTH)
# chess_engine.setposition(["e2e4"])
# print(chess_engine.bestmove()['move'])
# chess_engine.setposition(["e2e4", "e7e5"])
# print(chess_engine.bestmove()['move'])
# chess_engine.setposition(["e2e4", "e7e5", "d2d4"])
# print(chess_engine.bestmove()['move'])
# chess_engine.setposition(["e2e4", "e7e5", "d2d4", "d7d5"])
# print(chess_engine.bestmove()['move'])