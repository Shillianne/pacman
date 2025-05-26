import torch
import numpy as np
from net import PacmanNet
import os
from pacman_types import Seed
from util import manhattanDistance, nearestPoint, heat_maps
from game import Directions, Game, GameStateData, Grid
import random, util
# random.seed(42)  # For reproducibility
random.seed(Seed.get_value())
from game import Agent
from pacman import GameState
from multiAgents import MultiAgentSearchAgent
from typing import Optional
from line_profiler import profile
import pickle
import logging
logging.basicConfig(filename="logs/log_trans.log", level=logging.ERROR)


with open("vector_table.pickle", "rb") as pf:
    vector_table: dict[tuple[int, int], np.ndarray] = pickle.load(pf)


def custom_hash(state: GameState) -> int:
    return (hash(tuple(state.data.agentStates)) + 13*hash(state.data.food) + 113 * hash(tuple(state.data.capsules)) + 7 * hash(state.data.score))


def order_moves(moves: list[str], state: GameState, agentIndex: int):
    states: list[GameState] = []
    d = {}
    for i, move in enumerate(moves):
        guess_score = 0
        successor: GameState = state.generateSuccessor(agentIndex, move)
        states.append(successor)
        guess_score += successor.getNumFood() - state.getNumFood()
        if successor.isLose():
            guess_score -= 100
        guess_score += successor.getScore() - state.getScore()
        pos = successor.getPacmanPosition()
        if not any([g_state.scaredTimer > 8 for g_state in successor.getGhostStates()]):
            guess_score -= min(manhattanDistance(pos, ghost_pos) for ghost_pos in successor.getGhostPositions())
        else:
            guess_score += 10 * min(manhattanDistance(pos, ghost_pos) for ghost_pos in successor.getGhostPositions())
        d[i] = guess_score
    idxs: list[int] = sorted(list(d.keys()), key=lambda x: d[x], reverse=True)
    new_moves = [moves[idx] for idx in idxs]
    states = [states[idx] for idx in idxs]
    return new_moves, states


class Node:
    def __init__(self, name: str, hash: int, state: GameStateData, idx: int, alpha: float, beta: float, id: int):
        self.role = 'MAX' if idx == 0 else 'MIN'
        self.name = name
        self.hash = hash
        self.state = state
        self.children: list["Node"] = []
        self.bestmove: str = ""
        self.eval: float = 0.0
        self.id = id
        self.transpositioned = False
        self.alpha = alpha
        self.beta = beta
        self.alphabeta_history: list[tuple[float, float]] = []

    def __repr__(self):
        return f"Node<{self.role} | {self.name}>"


class Entry:
    def __init__(self, eval: float, eval_type: int, depth: int, hash: int, move: str):
        self.eval = eval
        self.eval_type = eval_type
        self.depth = depth
        self.hash = hash
        self.move = move


class TranspositionTable:
    LOOKUP_FAILED = False
    EXACT = 0
    LOWER_BOUND = 1
    UPPER_BOUND = 2
    def __init__(self):
        self.table: dict[int, Entry] = {}
        # self.table: dict[tuple[int, int], Entry] = {}
        
    def store_evaluation(self, key: int, ply: int, eval: float, eval_type: int, move: str):
        if key in self.table.keys():
            entry = self.table[key]
            if entry.depth < ply:
                self.table[key] = Entry(eval, eval_type, ply, key, move)
        else:
            self.table[key] = Entry(eval, eval_type, ply, key, move)

    def lookup_evaluation(self, key: int, ply: int, alpha: float, beta: float):
        if key in self.table.keys():
            entry = self.table[key]
            if entry.depth >= ply:
                if entry.eval_type == TranspositionTable.EXACT:
                    return entry.eval
                elif entry.eval_type == TranspositionTable.UPPER_BOUND and entry.eval <= alpha:
                    return entry.eval
                elif entry.eval_type == TranspositionTable.LOWER_BOUND and entry.eval >= beta:
                    return entry.eval
        return TranspositionTable.LOOKUP_FAILED


class SearchAgent(MultiAgentSearchAgent):
    def __init__(self, evalFn='customEvaluationFunction', depth=2,
                 alphabeta: bool | str = True,
                 transposition: bool | str = True,
                 ordering: bool | str = True,
                 layout: str = "mediumClassic") :
        super().__init__(evalFn, str(depth))
        self.bestmove = ""
        self.layout = layout
        self.best_eval: Optional[float] = None
        self.tree: Optional[Node] = None
        self.n_called = 0
        self.ply = self.depth * 3
        self.transpositionTable = TranspositionTable()
        self.logger = logging.getLogger(__name__)
        self.alphabeta = alphabeta if isinstance(alphabeta, bool) else alphabeta == "True"
        self.transposition = transposition if isinstance(transposition, bool) else transposition == "True"
        self.move_ordering =  ordering if isinstance(ordering, bool) else ordering == "True"
        self.ghosts_heat_map, self.current_heat_map, self.original_food  = heat_maps(self.layout)
        self.save = False

        print(f"Defined a Search Agent with a depth of {self.depth}, alphabeta {alphabeta}, transposition {transposition}, ordering {ordering} on map {self.layout}")
        if not self.alphabeta and not self.transposition and not self.move_ordering:
            self.file_ending = "minimax"
        elif self.alphabeta and not self.transposition and not self.move_ordering:
            self.file_ending = "alphabeta"
        elif self.alphabeta and self.transposition and not self.move_ordering:
            self.file_ending = "transposition"
        elif self.alphabeta and not self.transposition and self.move_ordering:
            self.file_ending = "ordering_notrans"
        elif self.alphabeta and self.transposition and self.move_ordering:
            self.file_ending = "ordering"
        else: raise ValueError(f"Cannot handle the inputed combination: alphabeta = {self.alphabeta}, transposition = {self.transposition}, ordering = {self.move_ordering}")
        self.condition_met = False
        self.identifier = 0

    @profile
    def search(self,
               agentIndex: int,
               alpha: float,
               beta: float,
               ply: int,
               state: GameState,
               root: Optional[Node] = None) -> float:
        # key = hash(state)
        key = custom_hash(state)
        if root is not None:
            root.alphabeta_history.append((alpha, beta))

        self.logger.debug(f"{'\t' * (self.ply - ply)}Current state hash: {key} | Current index: {agentIndex} | Current ply: {ply}")
        if state.isWin() or state.isLose() or ply == 0:
            self.logger.debug(f"{'\t' * (self.ply - ply)}Reached bottom of the search tree.")
            eval = self.evaluationFunction(self.ghosts_heat_map, self.current_heat_map, self.original_food, state)
            if root is not None:
                root.eval = eval
                root.alpha = alpha
                root.beta = beta
            return eval

        if self.transposition:
            possible_eval = self.transpositionTable.lookup_evaluation(key, ply, alpha, beta)
            if possible_eval != TranspositionTable.LOOKUP_FAILED:
                if root is not None:
                    root.transpositioned = True
                    root.eval = possible_eval
                return possible_eval

        moves = state.getLegalActions(agentIndex)
        if "Stop" in moves:
            moves.remove("Stop")
        if not moves:
            return self.evaluationFunction(self.ghosts_heat_map, self.current_heat_map, self.original_food, state)
        states = None
        if self.move_ordering:
            moves, states = order_moves(moves, state, agentIndex)
        next_agent = agentIndex + 1
        best_move: Optional[str] = None
        best_eval = float("-inf") if agentIndex == 0 else float("inf")

        # eval_bound = TranspositionTable.UPPER_BOUND
        o_alpha = alpha
        o_beta = beta

        self.logger.debug(f"{'\t' * (self.ply - ply)}Starting move search for state {key} and index {agentIndex}")
        for i, move in enumerate(moves):
            if self.move_ordering:
                assert states is not None
                successor = states[i]
            else:
                successor = state.generateSuccessor(agentIndex, move)
            s_key = custom_hash(successor)
            node = Node(move, s_key, successor.data, 0 if next_agent == state.getNumAgents() else next_agent, alpha, beta, self.identifier + 1) if root is not None else None
            self.identifier += 1
            self.logger.debug(f"{'\t' * (self.ply - ply)}Performing search for state {s_key} associated to move {move} and index {agentIndex}.")
            if agentIndex == 0:  # playing as pacman
                eval: float = self.search(agentIndex + 1, alpha, beta, ply - 1, successor, node)
            else:  # playing as a ghost
                eval: float = self.search(0 if next_agent == state.getNumAgents() else next_agent,
                                          alpha,
                                          beta,
                                          ply - 1,
                                          successor,
                                          node)
            # print(eval, best_eval)
            if agentIndex == 0:
                if eval > best_eval:
                    self.logger.debug(f"{'\t' * (self.ply - ply)}New best move for state {key}, index {agentIndex}, ply {ply}: {move}")
                    best_eval = eval
                    best_move = move
                    # eval_bound = TranspositionTable.EXACT
                    alpha = max(alpha, eval)
            else:
                if eval < best_eval:
                    self.logger.debug(f"{'\t' * (self.ply - ply)}New best move for state {key}, index {agentIndex}, ply {ply}: {move}")
                    best_eval = eval
                    best_move = move
                    # eval_bound = TranspositionTable.EXACT
                    beta = min(beta, eval)
            # if eval == 71 or self.condition_met:
            #     self.condition_met = True
            #     import code; code.interact(local=locals())
            if node is not None:
                node.alpha = alpha
                node.beta = beta
            if root is not None:
                root.alphabeta_history.append((alpha, beta))
            if self.alphabeta and beta <= alpha:
                    # if self.transposition:
                    #     self.transpositionTable.store_evaluation(s_key, ply, beta, TranspositionTable.LOWER_BOUND, move)
                    self.logger.debug(f"{'\t' * (self.ply - ply)}Prunning on state {key}, for index {agentIndex}, ply {ply}: ({alpha}, {beta})")
                    assert best_move is not None
                    if self.transposition:
                        if agentIndex == 0:  # Maximizing player - we have a lower bound
                            self.transpositionTable.store_evaluation(key, ply, best_eval, TranspositionTable.UPPER_BOUND, best_move)
                        else:  # Minimizing player - we have an upper bound
                            self.transpositionTable.store_evaluation(key, ply, best_eval, TranspositionTable.LOWER_BOUND, best_move)
                    if node is not None and root is not None:
                        # node.prunned = True
                        # root.children.append(node)
                        root.eval = beta if agentIndex == 0 else alpha
                    # if this happens, it means we found a move too good, so the oponnent will reject it for a better move for them
                    # return beta if agentIndex == 0 else alpha  # this is like saying: "Yeah, I got nothing better for me, so may as well say that I have nothing"
                    return best_eval

            if root is not None and node is not None:
                # save the explored search tree
                root.children.append(node)
                assert best_move is not None
                root.bestmove = best_move
                root.eval = best_eval

        assert best_move is not None, "Didn't find a move after performing search"
        if ply == self.ply:
            self.logger.debug(f"Exiting search having found bestmove: {best_move}")
            self.bestmove = best_move

        # if self.transposition:
        #     self.transpositionTable.store_evaluation(key, ply, best_eval, eval_bound, best_move)
        if self.transposition:
            if best_eval <= o_alpha:
                # All moves were worse than alpha - upper bound
                eval_bound = TranspositionTable.LOWER_BOUND
            elif best_eval > o_beta:
                # At least one move was better than beta - lower bound
                eval_bound = TranspositionTable.UPPER_BOUND
            else: 
                # Exact value
                eval_bound = TranspositionTable.EXACT
            self.transpositionTable.store_evaluation(key, ply, best_eval, eval_bound, best_move)

        return best_eval

    @profile
    def getAction(self, state: GameState):
        root = Node("root", custom_hash(state), state.data, 0, float("-inf"), float("inf"), 0)
        # root = None
        self.logger.info("Starting search")
        # print(self.n_called)
        # import code; code.interact(local=locals())
        eval = self.search(0, float("-inf"), float("inf"), self.ply, state, root)
        self.logger.info("Search finished")
        self.best_eval = eval
        if root is not None:
            with open(f"logs/tree_{self.n_called}_{self.file_ending}.pickle", "wb") as pf:
                pickle.dump(root, pf)
            self.n_called += 1
        return self.bestmove

    def final(self, state: GameState):
        if self.save and (state.isWin() or state.isLose()):
            with open("transposition_table.pickle", "wb") as pf:
                pickle.dump(self.transpositionTable, pf)
            print("Saved transposition table")
