# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from typing import Callable, cast
import torch
import math
import numpy as np
from net import PacmanNet
from copy import copy
import os
import pacman
from pacman_types import Number
from util import manhattanDistance
from game import Directions, Grid
import random, util
random.seed(42)  # For reproducibility
from game import Agent
from pacman import GameState
from line_profiler import profile
import pickle
import pandas as pd

with open("vector_table.pickle", "rb") as pf:
    vector_table: dict[tuple[int, int], np.ndarray] = pickle.load(pf)

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):  # type: ignore  # this method will never be extended properly
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

def providedEvaluationFunction(state: GameState):
    w1 = 0.0
    w2 = 0.0
    w3 = 0.0
    w4 = 0.0
    w5 = 0.0
    pacman_pos = state.getPacmanPosition()
    score = state.getScore()
    food_matrix = state.getFood()
    assert isinstance(food_matrix, Grid)
    food_distance = min(manhattanDistance(pacman_pos, (x, y)) for x in range(food_matrix.width) for y in range(food_matrix.height) if food_matrix[x][y])
    capsules_matrix = state.getCapsules()
    assert isinstance(capsules_matrix, Grid)
    capsule_distance= min(manhattanDistance(pacman_pos, (x, y)) for x in range(capsules_matrix.width) for y in range(capsules_matrix.height) if capsules_matrix[x][y])
    positions = state.getGhostPositions()
    ghost_distance = min(manhattanDistance(pacman_pos, pos) for pos in positions)
    scared_ghost_distance = 0
    ghost_state = [(i, g_state.scaredTimer > 0) for i, g_state in enumerate(state.getGhostStates())]
    if True in [s[1] for s in ghost_state]:
        scared_ghost_distance = min(manhattanDistance(pacman_pos, positions[idx]) for idx, isScared in ghost_state if isScared)
    return w1 * score + w2 * food_distance + w3 * capsule_distance + w4 * ghost_distance + w5 * scared_ghost_distance


def customEvaluationFunction(ghosts_heat_map: dict[tuple[int, ...], np.ndarray], current_heat_map: np.ndarray, original_food:list[int], state: GameState):

    # Taking object's coords 
    pacman_pos: tuple[Number, Number] = state.getPacmanPosition()
    pacman_pos = (int(pacman_pos[0]), int(pacman_pos[1]))
    ghosts_pos: list[tuple[Number, Number]] = state.getGhostPositions()
    ghosts_pos = [(int(ghost_pos[0]), int(ghost_pos[1])) for ghost_pos in ghosts_pos]

    # Updating current_heat_map 
    copied_heat_map = current_heat_map.copy()
    for ghost in ghosts_pos:
        copied_heat_map += ghosts_heat_map[tuple(map(int, ghost))]

    # Pacman's index o
    pos_eval: int = copied_heat_map[pacman_pos]

    # Taking the global score
    score = state.getScore()

    # Taking food-map
    food_map = state.getFood().copy()
    quadrants = util.divide_map(food_map)

    # Decide where the fuck pacman is
    where_is_pacman = util.where_am_i(pacman_pos, (food_map.height, food_map.width))

    # Obtaining current food proportion per quadrant 
    quadrant_food_proportion =  [np.sum(quadrants[i]) /original_food[i]for i in range(4)]
    current_proportion = quadrant_food_proportion[where_is_pacman]


    # Get centroids of the quadrants
    centroids = util.get_centroids(quadrants,(food_map.height, food_map.width))

    # Obtaning the nearest quadrant according to its centroid position 
    nearest_quadrant, nearest_centroid = nearest_quadrant(pacman_pos, where_is_pacman, centroids)
    

    f_current = current_proportion/centroids[where_is_pacman]**2
    f_nearest_centroid = 1/centroids[nearest_centroid]**2

    # Puntuacion total | Heat Map | Densidad Cuadrante 


    return score - (pos_eval * 2) + abs(f_current - f_nearest_centroid)


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction: Callable[[dict[tuple[int, ...], np.ndarray], np.ndarray, GameState], float] = util.lookup(evalFn, globals())
        self.depth = int(depth)

# class MinimaxAgent(MultiAgentSearchAgent):
#     """
#     Your minimax agent (question 2)
#     """

#     def getAction(self, gameState: GameState):
#         """
#         Returns the minimax action from the current gameState using self.depth
#         and self.evaluationFunction.

#         Here are some method calls that might be useful when implementing minimax.

#         gameState.getLegalActions(agentIndex):
#         Returns a list of legal actions for an agent
#         agentIndex=0 means Pacman, ghosts are >= 1

#         gameState.generateSuccessor(agentIndex, action):
#         Returns the successor game state after an agent takes an action

#         gameState.getNumAgents():
#         Returns the total number of agents in the game

#         gameState.isWin():
#         Returns whether or not the game state is a winning state

#         gameState.isLose():
#         Returns whether or not the game state is a losing state
#         """
#         "*** YOUR CODE HERE ***"
#         util.raiseNotDefined()

# class AlphaBetaAgent(MultiAgentSearchAgent):
#     """
#     Your minimax agent with alpha-beta pruning (question 3)
#     """

#     def getAction(self, gameState: GameState):
#         """
#         Returns the minimax action using self.depth and self.evaluationFunction
#         """
#         "*** YOUR CODE HERE ***"
#         util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):  # type: ignore  # this method will never be extended properly
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction


###########################################################################
# Ahmed
###########################################################################

class NeuralAgent(Agent):
    """
    Un agente de Pacman que utiliza una red neuronal para tomar decisiones
    basado en la evaluación del estado del juego.
    """
    def __init__(self, model_path="models/pacman_model.pth"):
        super().__init__()
        self.model = None
        self.input_size = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model(model_path)
        
        # Mapeo de índices a acciones
        self.idx_to_action = {
            0: Directions.STOP,
            1: Directions.NORTH,
            2: Directions.SOUTH,
            3: Directions.EAST,
            4: Directions.WEST
        }
        
        # Para evaluar alternativas
        self.action_to_idx = {v: k for k, v in self.idx_to_action.items()}
        
        # Contador de movimientos
        self.move_count = 0
        
        print(f"NeuralAgent inicializado, usando dispositivo: {self.device}")

    def load_model(self, model_path):
        """Carga el modelo desde el archivo guardado"""
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: No se encontró el modelo en {model_path}")
                return False
                
            # Cargar el modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            self.input_size = checkpoint['input_size']
            
            # Crear y cargar el modelo
            self.model = PacmanNet(self.input_size, 128, 5).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()  # Modo evaluación
            
            print(f"Modelo cargado correctamente desde {model_path}")
            print(f"Tamaño de entrada: {self.input_size}")
            return True
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            return False

    def state_to_matrix(self, state):
        """Convierte el estado del juego en una matriz numérica normalizada"""
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz numérica
        # 0: pared, 1: espacio vacío, 2: comida, 3: cápsula, 4: fantasma, 5: Pacman
        numeric_map = np.zeros((width, height), dtype=np.float32)
        
        # Establecer espacios vacíos (todo lo que no es pared comienza como espacio vacío)
        for x in range(width):
            for y in range(height):
                if not walls[x][y]:
                    numeric_map[x][y] = 1
        
        # Agregar comida
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    numeric_map[x][y] = 2
        
        # Agregar cápsulas
        for x, y in state.getCapsules():
            numeric_map[x][y] = 3
        
        # Agregar fantasmas
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            # Si el fantasma está asustado, marcarlo diferente
            if ghost_state.scaredTimer > 0:
                numeric_map[ghost_x][ghost_y] = 6  # Fantasma asustado
            else:
                numeric_map[ghost_x][ghost_y] = 4  # Fantasma normal
        
        # Agregar Pacman
        pacman_x, pacman_y = state.getPacmanPosition()
        numeric_map[int(pacman_x)][int(pacman_y)] = 5
        
        # Normalizar
        numeric_map = numeric_map / 6.0
        
        return numeric_map

    def evaluationFunction(self, state):
        """
        Una función de evaluación basada en la red neuronal y en heurísticas adicionales.
        """
        if self.model is None:
            return 0  # Si no hay modelo, devolver 0
        
        # Convertir a matriz
        state_matrix = self.state_to_matrix(state)
        
        # Convertir a tensor
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        # Obtener predicciones
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Aplicar heurísticas adicionales, similar a betterEvaluationFunction
        score = state.getScore()
        
        # Mejorar la evaluación con conocimiento del dominio
        pacman_pos = state.getPacmanPosition()
        food = state.getFood().asList()
        ghost_states = state.getGhostStates()
        
        # Factor 1: Distancia a la comida más cercana
        if food:
            min_food_distance = min(manhattanDistance(pacman_pos, food_pos) for food_pos in food)
            score += 1.0 / (min_food_distance + 1)
        
        # Factor 2: Proximidad a fantasmas
        for ghost_state in ghost_states:
            ghost_pos = ghost_state.getPosition()
            ghost_distance = manhattanDistance(pacman_pos, ghost_pos)
            
            if ghost_state.scaredTimer > 0:
                # Si el fantasma está asustado, acercarse a él
                score += 50 / (ghost_distance + 1)
            else:
                # Si no está asustado, evitarlo
                if ghost_distance <= 2:
                    score -= 200  # Gran penalización por estar demasiado cerca
        
        # Combinar la puntuación de la red con la heurística
        neural_score = 0
        for i, action in enumerate(self.idx_to_action.values()):
            if action in legal_actions:
                neural_score += probabilities[i] * 100
        
        return score + neural_score

    def getAction(self, state: GameState):  # type: ignore  # this method will never be extended properly
        """
        Devuelve la mejor acción basada en la evaluación de la red neuronal
        y heurísticas adicionales.
        """
        self.move_count += 1
        
        # Si no hay modelo, hacer un movimiento aleatorio
        if self.model is None:
            print("ERROR: Modelo no cargado. Haciendo movimiento aleatorio.")
            exit()
            legal_actions = state.getLegalActions()
            return random.choice(legal_actions)
        
        # Obtener acciones legales
        legal_actions = state.getLegalActions()
        
        # Evaluación directa con la red neuronal
        state_matrix = self.state_to_matrix(state)
        state_tensor = torch.FloatTensor(state_matrix).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            output = self.model(state_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1).cpu().numpy()[0]
        
        # Mapear índices del modelo a acciones del juego
        action_probs = []
        for idx, prob in enumerate(probabilities):
            action = self.idx_to_action[idx]
            if action in legal_actions:
                action_probs.append((action, prob))
        
        # Ordenar por probabilidad (mayor a menor)
        action_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Exploración: con una probabilidad decreciente, elegir aleatoriamente
        exploration_rate = 0.2 * (0.99 ** self.move_count)  # Disminuye con el tiempo
        if random.random() < exploration_rate:
            # Excluir STOP si es posible
            if len(legal_actions) > 1 and Directions.STOP in legal_actions:
                legal_actions.remove(Directions.STOP)
            return random.choice(legal_actions)
        
        # Evaluación alternativa: generar sucesores y evaluar cada uno
        successors = []
        for action in legal_actions:
            successor = state.generateSuccessor(0, action)
            eval_score = self.evaluationFunction(successor)
            neural_score = 0
            for a, p in action_probs:
                if a == action:
                    neural_score = p * 100
                    break
            # Combinar evaluación heurística con la predicción de la red
            combined_score = eval_score + neural_score
            
            # Penalizar STOP a menos que sea la única opción
            if action == Directions.STOP and len(legal_actions) > 1:
                combined_score -= 50
                
            successors.append((action, combined_score))
        
        # Ordenar por puntuación combinada
        successors.sort(key=lambda x: x[1], reverse=True)
        
        # Devolver la mejor acción
        return successors[0][0]


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Minimax agent for Pacman with multiple ghosts
    """

    @profile
    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction.
        """
        
        @profile
        def minimax(agentIndex, depth, gameState):
            """
            Recursive minimax function
            
            Args:
            - agentIndex: Current agent (0=Pacman, 1+=Ghosts)  
            - depth: Current depth in the game tree
            - gameState: Current state of the game
            
            Returns:
            - Best evaluation score for this state
            """
            # Base case: terminal state or maximum depth reached
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)  # type: ignore  # this class isn't used and the eval function is customized for another class

            # Pacman's turn (Maximizer)
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState)
            # Ghost's turn (Minimizer)  
            else:
                return minValue(agentIndex, depth, gameState)
        
        @profile
        def maxValue(agentIndex, depth, gameState):
            """
            Handles Pacman's moves (maximizing player)
            """
            v = float('-inf')  # Start with worst possible value
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(gameState)  # type: ignore  # this class isn't used and the eval function is customized for another class

            # Try each possible action and choose the best
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                # After Pacman moves, first ghost plays (agent 1)
                v = max(v, minimax(1, depth, successor))
            return v

        @profile
        def minValue(agentIndex, depth, gameState):
            """
            Handles Ghost moves (minimizing players)
            """
            v = float('inf')  # Start with best possible value for Pacman
            legalActions = gameState.getLegalActions(agentIndex)
            
            # No legal actions available
            if not legalActions:
                return self.evaluationFunction(gameState)  # type: ignore  # this class isn't used and the eval function is customized for another class

            # Determine next agent and depth
            nextAgent = agentIndex + 1
            nextDepth = depth
            
            # If all ghosts have moved, return to Pacman and increment depth
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0      # Back to Pacman
                nextDepth = depth + 1  # New ply begins

            # Try each possible action and choose the worst for Pacman
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, minimax(nextAgent, nextDepth, successor))
            return v

        # Main decision logic for Pacman
        bestAction = None
        bestScore = float('-inf')

        # Try each legal action for Pacman
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            # Start minimax with first ghost (agent 1) at current depth
            score = minimax(1, 0, successor)
            
            if score > bestScore:
                bestScore = score
                bestAction = action

        return bestAction


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Minimax agent with alpha-beta pruning
    """

    def getAction(self, gameState: GameState):  # type: ignore  # this method will never be extended properly
        """
        Returns the alpha-beta action using self.depth and self.evaluationFunction
        """
        
        def alphabeta(agentIndex, depth, gameState, alpha, beta):
            # Base case: Check if the game is over or if we've reached the maximum depth
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)  # type: ignore  # this class isn't used and the eval function is customized for another class

            # Pacman (maximizer) is agentIndex 0
            if agentIndex == 0:
                return maxValue(agentIndex, depth, gameState, alpha, beta)
            # Ghosts (minimizer) are agentIndex 1 or higher
            else:
                return minValue(agentIndex, depth, gameState, alpha, beta)

        def maxValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize max value
            v = float('-inf')
            # Get Pacman's legal actions
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)  # type: ignore  # this class isn't used and the eval function is customized for another class

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = max(v, alphabeta(1, depth, successor, alpha, beta))  # Ghosts start at index 1
                if v > beta:
                    return v  # Prune the remaining branches
                alpha = max(alpha, v)
            return v

        def minValue(agentIndex, depth, gameState, alpha, beta):
            # Initialize min value
            v = float('inf')
            # Get the current agent's legal actions (ghosts)
            legalActions = gameState.getLegalActions(agentIndex)

            if not legalActions:
                return self.evaluationFunction(gameState)  # type: ignore  # this class isn't used and the eval function is customized for another class

            # Get the next agent's index and check if we need to increase depth
            nextAgent = agentIndex + 1
            if nextAgent == gameState.getNumAgents():
                nextAgent = 0  # Go back to Pacman
                depth += 1  # Increase the depth since we've gone through all agents

            # Iterate through all possible actions and update alpha-beta values
            for action in legalActions:
                successor = gameState.generateSuccessor(agentIndex, action)
                v = min(v, alphabeta(nextAgent, depth, successor, alpha, beta))
                if v < alpha:
                    return v  # Prune the remaining branches
                beta = min(beta, v)
            return v

        # Pacman (agentIndex 0) will choose the action with the best alpha-beta score
        bestAction = None
        bestScore = float('-inf')
        alpha = float('-inf')
        beta = float('inf')

        for action in gameState.getLegalActions(0):  # Pacman's legal actions
            successor = gameState.generateSuccessor(0, action)
            score = alphabeta(1, 0, successor, alpha, beta)  # Start with Ghost 1, depth 0
            if score > bestScore:
                bestScore = score
                bestAction = action
            alpha = max(alpha, score)

        return bestAction

# Definir una función para crear el agente
def createNeuralAgent(model_path="models/pacman_model.pth"):
    """
    Función de fábrica para crear un agente neuronal.
    Útil para integrarse con la estructura de pacman.py.
    """
    return NeuralAgent(model_path)
