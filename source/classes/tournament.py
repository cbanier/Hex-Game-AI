import os
import pickle
import logging
from rich import print
from rich.logging import RichHandler

FORMAT = "%(message)s"
logging.basicConfig(level="NOTSET", format=FORMAT, datefmt="[%X]", handlers=[RichHandler()])

# Hide Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame

from classes.game import Game

import pandas as pd

from classes.strategy import play_move_time

import time

class Tournament:
    def __init__(self, args:  list):
        """
        Initialises a tournament with:
           * the size of the board,
           * the playing mode, i.e., "ai_vs_ai", "man_vs_ai",
           * the game counter,
           * the number of games to play.
        """
        self.args = args
        self.BOARD_SIZE = args[0]
        self.MODE = args[1]
        self.GAME_COUNT = args[2]
        self.N_GAMES = args[3]

    def single_game(self, black_starts: bool = True) -> int:
        """
        Runs a single game between two opponents.

        @return   The number of the winner, either 1 or 2, for black and white respectively.
        """

        pygame.init()
        pygame.display.set_caption("Polyline")

        game = Game(board_size = self.BOARD_SIZE, mode = self.MODE, black_starts = black_starts)
        game.get_game_info([ self.BOARD_SIZE, self.MODE, self.GAME_COUNT ])
        
        while game.winner is None:
            game.play()

        print(f"\nStatistiques :\n\nNumber of turns played : {game.nb_turns}")
        print(f"Number of moves played by black : {game.nb_turns}") # black's player plays first so his number of moves is always equal to the number of turns, no matter what the outcome

        # The number of white moves is equal to the number of turns played if white wins, otherwise it is equal to the number of turns minus 1 as blacks always start
        if game.winner == 1: # if black wins
            value = game.nb_turns -1
        else:
            value = game.nb_turns

        print(f"Number of moves played by white : {value}") 
         
        return game.winner

    def championship(self):
        """
        Runs a number of games between the same two opponents.
        """
        for _ in range(self.N_GAMES):
            self.GAME_COUNT = _

            # First half of the tournament started by one player.
            # Remaining half started by other player (see "no pie rule")
            winner = self.single_game(black_starts = self.GAME_COUNT < self.N_GAMES / 2)


        log = logging.getLogger("rich")

        # creation of a dictionary to count the victories of each player
        win_count = {1 : 0, 2 : 0}
        for _ in range(self.N_GAMES):
            self.GAME_COUNT = _

            winner = self.single_game(black_starts = self.GAME_COUNT < self.N_GAMES / 2)
            # The winner of each game is recorded by incrementing the corresponding value in the "win_count" dictionary.
            win_count[winner] += 1

        print(f"\nBlack Player won {int(win_count[1])} games || White Player won {int(win_count[2])} games")
        print(f"Win rate Black player: {int((win_count[1]/self.N_GAMES)*100)}% || Win rate White player: {int((win_count[2]/self.N_GAMES)*100)}% \n")

        # Average time for each player to play a move

        print(f"Black player took an average of {sum(play_move_time[1])/len(play_move_time) * 1000} milliseconds to make a move during these games")
        print(f"White player took an average of {sum(play_move_time[2])/len(play_move_time) * 1000} milliseconds to make a move during these games\n")