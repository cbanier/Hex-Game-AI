import copy
from math import log, sqrt, inf, floor
from random import choice
from typing import Optional

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from classes.utils import index_finder, all_equal

class Node(object):
    def __init__(self, logic, board, move=(None, None), wins=0, visits=0, children=None):
        # Save the #wins:#visited ratio
        self.state = board
        self.move = move
        self.wins = wins
        self.visits = visits
        self.children = children or []
        self.parent = None
        self.untried_moves = logic.get_possible_moves(board)

    def __str__(self, level=0):
        ret = "\t"*level+repr(self.move)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret
    
    def __repr__(self):
        return '<tree node representation>'

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


class STRAT:
    def __init__(self, logic, ui, board_state, starting_player):
        self.logic = logic
        self.ui = ui
        self.root_state = copy.copy(board_state)
        self.state = copy.copy(board_state)
        self.starting_player = starting_player
        self.players = [1, 2]
        self.players.remove(self.starting_player)
        self.other_player = self.players[0]
        self.turn = {True: self.starting_player, False: self.other_player}
        self.turn_state = True

    def start(self) -> tuple:
        root_node = Node(self.logic, self.root_state)

        if self.starting_player is self.ui.BLACK_PLAYER:
            # implement here Black player strategy (if needed, i.e., no human playing)
            x, y = self.random_strategy(root_node)

        elif self.starting_player is self.ui.WHITE_PLAYER:
            # implement here White player strategy
            #x, y = self.minimax_strategy(root_node)
            x, y = self.minimax_alpha_beta_strategy(root_node)

        return (x, y)

    ##################################################
    #                RANDOM STRATEGY                 #
    ##################################################

    def random_strategy(self, node: Node) -> tuple:
        return choice(node.untried_moves)

    ##################################################
    #              AUXILIARY FUNCTIONS               #
    ##################################################

    def create_children(self, current_node: Node, player: int):
        for x, y in current_node.untried_moves:
            new_state = np.copy(current_node.state)
            new_state[x][y] = player
            new_node = Node(self.logic, board=new_state, move=(x, y))
            current_node.add_child(new_node)

    ##################################################
    #                   HEURISTICS                   #
    ##################################################
    
    def explore_board_horizontally(self):
        pass


    def naive_heuristic(self, board: np.ndarray) -> int:
        if self.is_winning_path(self.starting_player, board) is self.starting_player:
            return 1 if self.starting_player is self.ui.BLACK_PLAYER else -1
        
        # we have to explore the grid, in order to understand which player is more able to win the game
        else:
            for i in range(self.ui.board_size):
                for j in range(self.ui.board_size):
                    pass
            print(board)
            return 1

    ##################################################
    #                    MINIMAX                     #
    ##################################################

    def minimax_aux(self, current_node: Node, player: int) -> int:
        if self.logic.is_game_over(self.ui.WHITE_PLAYER, current_node.state) is self.ui.WHITE_PLAYER:
            self.logic.GAME_OVER = False
            return -1
        elif self.logic.is_game_over(self.ui.BLACK_PLAYER, current_node.state) is self.ui.BLACK_PLAYER:
            self.logic.GAME_OVER = False
            return 1

        self.create_children(current_node, player)

        if player is self.ui.BLACK_PLAYER:
            value = -inf
            for child in current_node.children:
                value = max(value, self.minimax_aux(child, self.ui.WHITE_PLAYER))

        else:
            value = inf
            for child in current_node.children:
                value = min(value, self.minimax_aux(child, self.ui.BLACK_PLAYER))

        return value


    def minimax_strategy(self, root: Node) -> tuple:
        self.create_children(root, self.starting_player)

        values = []
        for child in root.children:
            values.append(self.minimax_aux(child, self.other_player))

        if all_equal(values):
            best_move = choice(root.children).move

        else:
            best_value = max(values) if self.starting_player is self.ui.BLACK_PLAYER else min(values)
            best_move = root.children[choice(index_finder(values, best_value))].move

            print(best_move)
        return best_move

    ##################################################
    #               MINIMAX ALPHA BETA               #
    ##################################################

    def minimax_alpha_beta_aux(self, current_node: Node, player: int, alpha: int, beta: int, depth: int) -> int:
        if self.logic.is_game_over(self.ui.WHITE_PLAYER, current_node.state) is self.ui.WHITE_PLAYER:
            self.logic.GAME_OVER = False
            return -1
        elif self.logic.is_game_over(self.ui.BLACK_PLAYER, current_node.state) is self.ui.BLACK_PLAYER:
            self.logic.GAME_OVER = False
            return 1
        elif depth == 0:
            return 0
        
        self.create_children(current_node, player)

        if player is self.ui.BLACK_PLAYER:
            value = -inf
            for child in current_node.children:
                value = max(value, self.minimax_alpha_beta_aux(child, self.ui.WHITE_PLAYER, alpha, beta, depth - 1))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = inf
            for child in current_node.children:
                value = min(value, self.minimax_alpha_beta_aux(child, self.ui.BLACK_PLAYER, alpha, beta, depth - 1))
                beta = min(beta, value)
                if beta <= alpha:
                    break

        return value


    def minimax_alpha_beta_strategy(self, root: Node, alpha: int = -2, beta: int = 2) -> tuple:
        self.create_children(root, self.starting_player)
        
        #mean_turn = 3 * floor(self.ui.board_size / 2)

        values = []
        for child in root.children:
            value = self.minimax_alpha_beta_aux(child, self.other_player, alpha, beta, depth=4)
            print(f"val : {value} ; move : {child.move}")
            values.append(value)
        
        if all_equal(values):
            best_move = choice(root.children).move

        else:
            best_value = max(values) if self.starting_player is self.ui.BLACK_PLAYER else min(values)
            best_move = root.children[choice(index_finder(values, best_value))].move

        print(best_move)
        return best_move