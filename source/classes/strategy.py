import copy
from math import log, sqrt, inf, floor
from random import choice
from typing import List, Optional

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table

from classes.utils import index_finder, all_equal

import time

play_move_time = {1 : [], 2 : []}

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

    ##################################################
    #             TREE SEARCH FUNCTIONS              #
    ##################################################

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.move) + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret
    

    def __repr__(self):
        return '<Tree node representation>'


    def add_child(self, child):
        child.parent = self
        self.children.append(child)


    def create_children(self, logic, player: int, moves_heuritic: bool = False):
        if moves_heuritic:
            for x, y in self.get_moves_to_explore(logic, player):
                new_state = np.copy(self.state)
                new_state[x][y] = player
                new_node = Node(logic, board=new_state, move=(x, y))
                self.add_child(new_node)
        else:
            for x, y in self.untried_moves:
                new_state = np.copy(self.state)
                new_state[x][y] = player
                new_node = Node(logic, board=new_state, move=(x, y))
                self.add_child(new_node)
        
    ##################################################
    #        HEURISTICS ABOUT MOVE EXPLORATION       #
    ##################################################

    def get_moves_to_explore(self, logic, player: int):
        board = np.copy(self.state)
        
        moves_of_player, moves_of_other_player = [], []
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == player:
                    moves_of_player.append((x, y))

                elif board[x][y] not in [player, 0]:
                    moves_of_other_player.append((x, y))

        moves_to_explore = []
        # Get free moves which are neighbors to the adversary player
        for move in moves_of_other_player:
            for neighbour in logic.get_neighbours(move):
                if (neighbour not in moves_to_explore and neighbour not in moves_of_other_player
                    and neighbour not in moves_of_player):
                    moves_to_explore.append(neighbour)

        # adding a path allowing the player to win by moving respectively along the y-axis for the white player or the x-axis for the black player
        others_moves_to_explore = []
        if player == 1: # BLACK_PLAYER
            for x, y in moves_to_explore:
                for delta_x in range(len(board)):
                    if logic.is_node_free((delta_x, y), board) == True and (delta_x, y) not in others_moves_to_explore:
                        others_moves_to_explore.append((delta_x, y))
                        board[delta_x][y] = player

            for x, y in moves_of_player:
                for delta_y in range(len(board)):
                    if logic.is_node_free((x, delta_y), board) == True and (x, delta_y) not in others_moves_to_explore:
                        others_moves_to_explore.append((x, delta_y))
                        board[x][delta_y] = player

        else:
            for x, y in moves_to_explore:
                for delta_y in range(len(self.state)):
                    if logic.is_node_free((x, delta_y), board) == True and (x, delta_y) not in others_moves_to_explore:
                        others_moves_to_explore.append((x, delta_y))
                        board[x][delta_y] = player

            for x, y in moves_of_player:
                for delta_x in range(len(self.state)):
                    if logic.is_node_free((delta_x, y), board) == True and (delta_x, y) not in others_moves_to_explore:
                        others_moves_to_explore.append((delta_x, y))
                        board[delta_x][y] = player

        for move in others_moves_to_explore:
            if move not in moves_to_explore:
                moves_to_explore.append(move)

        return moves_to_explore


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

        start_time = time.time()

        if self.starting_player is self.ui.BLACK_PLAYER:
            # implement here Black player strategy (if needed, i.e., no human playing)
            x, y = self.random_strategy(root_node)

        elif self.starting_player is self.ui.WHITE_PLAYER:
            # implement here White player strategy
            # x, y = self.minimax_strategy(root_node)
            # x, y = self.minimaxAB_strategy(root_node)
            x, y = self.minimaxAB_bestChoice(root_node)
            print(f"({x}, {y})\n")

        time_elapsed = time.time() - start_time
        play_move_time[self.starting_player].append(time_elapsed)
        
        return (x, y)

    ##################################################
    #                RANDOM STRATEGY                 #
    ##################################################

    def random_strategy(self, node: Node) -> tuple:
        return choice(node.untried_moves)

    ##################################################
    #              AUXILIARY FUNCTIONS               #
    ##################################################

    def first_move_choose(self, player: int):
        board_size = len(self.root_state)
    
        top_part, bottom_part = [], []
        left_part, right_part = [], []
        for x in range(board_size):
            for y in range(x, board_size - x):
                top_part.append((x, y))
                left_part.append((y, x))

            for y in range(board_size - 1 - x, x + 1):
                bottom_part.append((x, y))
                right_part.append((y, x))

        if player is self.ui.BLACK_PLAYER:
            return choice(left_part + [elem for elem in right_part if elem not in left_part])
        return choice(top_part + [elem for elem in bottom_part if elem not in top_part])


    def get_score(self, board: np.array, player: int, argc: int) -> tuple:
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

        return None


    def choose_best_move(self, current_node: Node, minimax_values, path_lengths, depths) -> tuple:
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
                # print(f"depth_indexes : {depth_indexes} ; path_indexes : {path_indexes} ; inter : {inter_depth_path}")

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
            # print(f"depth_indexes : {depth_indexes} ; path_indexes : {path_indexes} ; inter : {inter_minimax_path} ; inter2 : {inter_minimax_depth}")
            inter_of_inters = [elem for elem in inter_minimax_path if elem in inter_minimax_depth]

            if len(inter_of_inters) > 0:
                return current_node.children[choice(inter_of_inters)].move
            elif len(inter_minimax_depth) > 0:
                return current_node.children[choice(inter_minimax_depth)].move
            elif len(inter_minimax_path) > 0:
                return current_node.children[choice(inter_minimax_path)].move
            return current_node.children[choice(minimax_indexes)].move

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
    #          MINIMAX ALPHA BETA OPTIMIZED          #
    ##################################################

    def minimaxAB_bestChoice_aux(self, current_node: Node, player: int, alpha: int, beta: int, depth: int) -> tuple:
        score = self.get_score(current_node.state, player, argc=2)
        if score is not None:
            score_minimax, path_length = score
            return (score_minimax, path_length, depth)

        if depth == 0:
            if player is self.starting_player:
                return (1, inf, -inf) if player is self.ui.BLACK_PLAYER else (-1, inf, -inf)
            return (-1, inf, -inf) if player is self.ui.BLACK_PLAYER else (1, inf, -inf)

        current_node.create_children(self.logic, player, moves_heuritic=True)

        best_path_length, best_depth = inf, -inf
        if player is self.ui.BLACK_PLAYER:
            best_value_minimax  = -inf
            for child in current_node.children:
                value_minimax, path_length, depth_acc = self.minimaxAB_bestChoice_aux(child, self.ui.WHITE_PLAYER, alpha, beta, depth - 1)
                # The mimimum path is the path which cross straight the board
                # Therefore, the minimum size of the path is the board_size
                if path_length >= len(self.root_state):
                    best_path_length = min(path_length, best_path_length)
                
                best_depth = max(depth_acc, best_depth)

                best_value_minimax = max(value_minimax, best_value_minimax)
                alpha = max(alpha, best_value_minimax)
                if beta <= alpha:
                    break
        else:
            best_value_minimax = inf
            for child in current_node.children:
                value_minimax, path_length, depth_acc = self.minimaxAB_bestChoice_aux(child, self.ui.BLACK_PLAYER, alpha, beta, depth - 1)
                if path_length >= len(self.root_state):
                    best_path_length = min(path_length, best_path_length)
                
                best_depth = max(depth_acc, best_depth)

                best_value_minimax = min(value_minimax, best_value_minimax)
                beta = min(beta, best_value_minimax)
                if beta <= alpha:
                    break

        return (best_value_minimax, best_path_length, best_depth)


    def minimaxAB_bestChoice(self, root: Node, alpha: int = -2, beta: int = 2, depth: int = 4) -> tuple:
        # Test if the board game is empty
        # i.e if the number of possible moves is equal to the dimension of the game
        if len(root.untried_moves) == len(self.root_state) ** 2:
            return self.first_move_choose(self.starting_player)

        root.create_children(self.logic, self.starting_player, moves_heuritic=True)

        minimax_values, path_lengths, depths = [], [], []
        for child in root.children:
            value, path_length, depth_acc = self.minimaxAB_bestChoice_aux(child, self.other_player, alpha, beta, depth=depth)
            #print(f"val : {value} ; move : {child.move} ; path_length : {path_length} ; depth : {depth_acc}")
            minimax_values.append(value)
            path_lengths.append(path_length)
            depths.append(depth_acc)

        return self.choose_best_move(root, minimax_values, path_lengths, depths)