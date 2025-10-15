from easyAI import Human_Player, TwoPlayerGame, AI_Player, Negamax

"""
https://en.wikipedia.org/wiki/Nim
https://zulko.github.io/easyAI/installation.html
Filip Patuła s28615, Michał Bedra s28854
"""

class Nim(TwoPlayerGame):
    """ In turn, the players remove from one to any number of sticks from selected pile.
    The player who removes the last stick loses. """
    def __init__(self, players):
        self.players = players
        self.current_player = 1
        self.piles = [1, 3, 5, 7, 9]

    def possible_moves(self):
        """ Generate list of possible moves, where the first number in move object is a pile number
            and the second number in move object is a number of sticks to take from pile
            Parameters:
            self
            Returns:
            [(int, int)]: list of moves, where each move is a tuple (x, y), where x is the pile number and y is the number of sticks to take from pile
        """
        moves = []
        for i in range(0, len(self.piles)):
            take_move_limit = self.piles[i] + 1
            for j in range(1, take_move_limit):
                moves.append((i, j))
        return moves

    def make_move(self, move):
        """ Removes a number of sticks from the selected pile
            Parameters:
            self,
            move (int, int): selected pile, number of sticks to take from pile

            Returns:
            None
         """
        self.piles[move[0]] -= move[1]

    def win(self):
        """Checks if piles are empty

           Parameters:
           self

           Returns:
           bool: True if piles are empty, False otherwise
          """
        for i in range(0, len(self.piles)):
            if self.piles[i] != 0:
                return False
        return True

    def is_over(self):
        """ Determine end state
            Parameters:
            self

            Returns:
            bool: True if win condition, False otherwise
        """
        return self.win()

    def show(self):
        """ Shows state of the Nim game
            Parameters:
            self

            Returns:
            None
        """
        for i in range(0, len(self.piles)):
            print("Pile " + str(i) + ": ")
            print(self.piles[i] * "|")

    def scoring(self):
        """ Scoring of the state of the Nim game for AI
            Parameters:
            self

            Returns:
            1 if win condition, 0 otherwise
        """
        return 1 if self.win() else 0

ai_player = AI_Player(Negamax(5))
game = Nim(players=[Human_Player(), ai_player])
history = game.play()
print("Player {} won!".format(game.current_player))