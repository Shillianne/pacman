import csv
import os
from game import Agent
import random
from game import Directions
import json
from datetime import datetime

from pacman_types import Seed
class CSVPlaybackAgent(Agent):
    """
    Un agente que reproduce un juego desde datos guardados en CSV
    """
    
    def __init__(self, csv_file_path):
        super().__init__(index=0)
        self.actions = []
        self.current_step = 0
        self.maps = []  # Guardar los mapas para visualización/depuración
        self.load_actions_from_csv(csv_file_path)
        # random.seed(42)  # Para reproducibilidad
        random.seed(Seed.get_value())
    
    def load_actions_from_csv(self, csv_file_path):
        print(f"Cargando acciones desde: {csv_file_path}")
        
        with open(csv_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if int(row['agent_index']) == 0:  # Solo Pacman
                    self.actions.append(row['action'])
                    # También podemos cargar los mapas si los necesitamos
                    if 'map_matrix' in row:
                        self.maps.append(json.loads(row['map_matrix']))
        
        print(f"Cargadas {len(self.actions)} acciones")
    
    def getAction(self, state):
        """Devuelve la siguiente acción del CSV"""
        if self.current_step < len(self.actions):
            action = self.actions[self.current_step]
            self.current_step += 1
            
            # Asegurarse de que la acción sea válida
            legal_actions = state.getLegalActions()
            if action in legal_actions:
                return action
            else:
                # Si la acción no es legal, elegir una legal aleatoriamente
                print(f"Warning: Acción {action} no es legal. Eligiendo aleatoriamente.")
                return random.choice(legal_actions)
        else:
            # Si se acaban las acciones, quedarse quieto
            return Directions.STOP
