import copy
from math import log, sqrt, inf, floor
from random import choice
from typing import List, Optional

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

    def create_children(self, logic, player: int):
        for x, y in self.untried_moves:
            new_state = np.copy(self.state)
            new_state[x][y] = player
            new_node = Node(logic, board=new_state, move=(x, y))
            self.add_child(new_node)
        
    # Distance between two Node
    def manhattan_distance(self, node):
        x1, y1 = self.move
        x2, y2 = node
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, node):
        x1, y1 = self.move
        x2, y2 = node
        return sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

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
            # x, y = self.minimax_strategy(root_node)
            # x, y = self.minimaxAB_strategy(root_node)
            x, y = self.minimaxAB_bestChoice(root_node)
            print(f"({x}, {y})\n")

        return (x, y)

    ##################################################
    #                RANDOM STRATEGY                 #
    ##################################################

    def random_strategy(self, node: Node) -> tuple:
        return choice(node.untried_moves)

    ##################################################
    #              AUXILIARY FUNCTIONS               #
    ##################################################

    def get_score(self, board: np.array, player: int, argc: int):
        path = self.logic.is_game_over(player, board)
        if path is not None:
            self.logic.GAME_OVER = False
            if path["player"] is player:
                if argc == 1:
                    return 1 if player is self.ui.BLACK_PLAYER else -1
                return (1, len(path["nodes"])) if player is self.ui.BLACK_PLAYER else (-1, len(path["nodes"]))

        other_player = self.ui.BLACK_PLAYER if player is self.ui.WHITE_PLAYER else self.ui.WHITE_PLAYER
        path = self.logic.is_game_over(other_player, board)
        if path is not None:
            self.logic.GAME_OVER = False
            if path["player"] is other_player:
                if argc == 1:
                    return -1 if player is self.ui.BLACK_PLAYER else 1
                return (-1, len(path["nodes"])) if player is self.ui.BLACK_PLAYER else (1, len(path["nodes"]))


    def make_best_move(self, current_node: Node, minimax_values, path_lengths, depths):
        best_path_length = min(path_lengths)
        best_depth = max(depths)

        """Summary of variables used below

        path_indexes    : Get indexes in path_lengths which correspond to best_path_length
        depth_indexes   : Get indexes in depths which correspond to best_depth
        minimax_indexes : Get indexes in minimax_values which correspond to the best_value, i.e min or max

        inter_depth_path    : Determine the intersection between path's and depth's indexes
        inter_minimax_path  : Determine the intersection between minimax's and path's indexes
        inter_minimax_depth : Determine the intersection between minimax's and depth's indexes
        inter_of_inters     : Determine the intersection between inter_minimax_path and inter_minimax_depth
        """

        if all_equal(minimax_values):
            path_indexes = index_finder(path_lengths, best_path_length)
            if all_equal(depths):
                # Pick a move which minimize the path length
                return current_node.children[choice(path_indexes)].move
            else:
                depth_indexes = index_finder(depths, best_depth)
                inter_depth_path = [elem for elem in depth_indexes if elem in path_indexes]
                print(f"depth_indexes : {depth_indexes} ; path_indexes : {path_indexes} ; inter : {inter_depth_path}")
                # If the intersection isn't empty then return one of these indexes
                # else return one of the path_indexes
                return current_node.children[choice(inter_depth_path)].move if len(inter_depth_path) > 0 else current_node.children[choice(path_indexes)].move

        else:
            best_value = max(minimax_values) if self.starting_player is self.ui.BLACK_PLAYER else min(minimax_values)

            minimax_indexes = index_finder(minimax_values, best_value)
            path_indexes = index_finder(path_lengths, best_path_length) 
            depth_indexes = index_finder(depths, best_depth)

            inter_minimax_path = [elem for elem in minimax_indexes if elem in path_indexes]
            inter_minimax_depth = [elem for elem in minimax_indexes if elem in depth_indexes]
            print(f"depth_indexes : {depth_indexes} ; path_indexes : {path_indexes} ; inter : {inter_minimax_path} ; inter2 : {inter_minimax_depth}")
            inter_of_inters = [elem for elem in inter_minimax_path if elem in inter_minimax_depth]

            if len(inter_of_inters) > 0:
                return current_node.children[choice(inter_of_inters)].move
            elif len(inter_minimax_depth) > 0:
                return current_node.children[choice(inter_minimax_depth)].move
            elif len(inter_minimax_path) > 0:
                return current_node.children[choice(inter_minimax_path)].move
            return current_node.children[choice(minimax_indexes)].move

    ##################################################
    #                   HEURISTICS                   #
    ##################################################

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

        current_node.create_children(self.logic, player)

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
        root.create_children(self.logic, self.starting_player)

        minimax_values = []
        for child in root.children:
            minimax_values.append(self.minimax_aux(child, self.other_player))

        if all_equal(minimax_values):
            best_move = choice(root.children).move
        else:
            best_value = max(minimax_values) if self.starting_player is self.ui.BLACK_PLAYER else min(minimax_values)
            best_move = root.children[choice(index_finder(minimax_values, best_value))].move
        
        return best_move

    ##################################################
    #               MINIMAX ALPHA BETA               #
    ##################################################

    def minimaxAB_aux(self, current_node: Node, player: int, alpha: int, beta: int, depth: int) -> int:
        score = self.get_score(current_node.state, player, argc=1)
        if score is not None:
            return score

        if depth == 0:
            if player is self.starting_player:
                return 1 if player is self.ui.BLACK_PLAYER else -1
            return -1 if player is self.ui.BLACK_PLAYER else 1
        
        current_node.create_children(self.logic, player)

        if player is self.ui.BLACK_PLAYER:
            value = -inf
            for child in current_node.children:
                value = max(value, self.minimaxAB_aux(child, self.ui.WHITE_PLAYER, alpha, beta, depth - 1))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
        else:
            value = inf
            for child in current_node.children:
                value = min(value, self.minimaxAB_aux(child, self.ui.BLACK_PLAYER, alpha, beta, depth - 1))
                beta = min(beta, value)
                if beta <= alpha:
                    break

        return value


    def minimaxAB_strategy(self, root: Node, alpha: int = -2, beta: int = 2, depth: int = 4) -> tuple:
        root.create_children(self.logic, self.starting_player)

        minimax_values = []
        for child in root.children:
            value = self.minimaxAB_aux(child, self.other_player, alpha, beta, depth)
            minimax_values.append(value)

        if all_equal(minimax_values):
            best_move = choice(root.children).move

        else:
            best_value = max(minimax_values) if self.starting_player is self.ui.BLACK_PLAYER else min(minimax_values)
            best_move = root.children[choice(index_finder(minimax_values, best_value))].move

        return best_move

    ##################################################
    #           PIMP MY MINIMAX ALPHA BETA           #
    ##################################################

    def minimaxAB_bestChoice_aux(self, current_node: Node, player: int, alpha: int, beta: int, depth: int) -> tuple:
        score = self.get_score(current_node.state, player, argc=2)
        if score is not None:
            score_minimax, path_length = score
            return (score_minimax, path_length, depth)

        if depth == 0:
            if player is self.starting_player:
                return (1, inf, depth) if player is self.ui.BLACK_PLAYER else (-1, inf, depth)
            return (-1, inf, depth) if player is self.ui.BLACK_PLAYER else (1, inf, depth)

        current_node.create_children(self.logic, player)

        if player is self.ui.BLACK_PLAYER:
            best_value_minimax, best_path_length = -inf, inf
            for child in current_node.children:
                value_minimax, path_length, depth_acc = self.minimaxAB_bestChoice_aux(child, self.ui.WHITE_PLAYER, alpha, beta, depth - 1)
                # The mimimum path is the path which cross straight the board
                # Therefore, the minimum size of the path is the board_size
                if path_length >= self.ui.board_size:
                    best_path_length = min(path_length, best_path_length)
                
                best_value_minimax = max(value_minimax, best_value_minimax)
                alpha = max(alpha, best_value_minimax)
                if beta <= alpha:
                    break
        else:
            best_value_minimax, best_path_length = inf, inf
            for child in current_node.children:
                value_minimax, path_length, depth_acc = self.minimaxAB_bestChoice_aux(child, self.ui.BLACK_PLAYER, alpha, beta, depth - 1)
                if path_length >= self.ui.board_size:
                    best_path_length = min(path_length, best_path_length)
                
                best_value_minimax = min(value_minimax, best_value_minimax)
                beta = min(beta, best_value_minimax)
                if beta <= alpha:
                    break

        return (best_value_minimax, best_path_length, depth_acc)


    def minimaxAB_bestChoice(self, root: Node, alpha: int = -2, beta: int = 2, depth: int = 4) -> tuple:
        root.create_children(self.logic, self.starting_player)

        minimax_values, path_lengths, depths = [], [], []
        for child in root.children:
            value, path_length, depth_acc = self.minimaxAB_bestChoice_aux(child, self.other_player, alpha, beta, depth=depth)
            print(f"val : {value} ; move : {child.move} ; path_length : {path_length} ; depth : {depth_acc}")
            minimax_values.append(value)
            path_lengths.append(path_length)
            depths.append(depth_acc)

        return self.make_best_move(root, minimax_values, path_lengths, depths)