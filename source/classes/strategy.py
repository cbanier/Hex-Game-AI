import copy
from math import log, sqrt, inf
from random import choice

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table


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
            x, y = self.random_strategy()

        elif self.starting_player is self.ui.WHITE_PLAYER:
            # implement here White player strategy
            x, y = self.minimax_strategy()

        return (x, y)

    def random_strategy(self) -> tuple:
        root_node = Node(self.logic, self.root_state)
        return choice(root_node.untried_moves)

    def minimax_strategy(self) -> tuple:
        self.tree() # call to test

        return self.random_strategy()

    def tree_builder(self, current_node, depth, player):
        if depth > 0 and current_node.untried_moves != []:
            for x, y in current_node.untried_moves:
                new_state = copy.copy(current_node.state)
                #print(f"av --> {new_state}")
                new_state[x][y] = player
                #print(f"ap --> {new_state}")

                new_node = Node(self.logic, board=new_state, move=(x, y))
                current_node.add_child(new_node)

            for child in current_node.children:
                if player == self.ui.BLACK_PLAYER:
                    self.tree_builder(child, depth - 1, self.ui.WHITE_PLAYER)
                else:
                    self.tree_builder(child, depth - 1, self.ui.BLACK_PLAYER)


    def tree(self):
        root = Node(self.logic, self.root_state)
        max_depth = 3 # max depth that we want to explore
        self.tree_builder(root, max_depth, self.starting_player)

        print(root)