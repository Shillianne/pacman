import csv
import json
import os
from datetime import datetime

class GameDataCollector:
    def __init__(self, output_dir="pacman_data", replay_mode=False):
        self.current_game_data = []
        self.output_dir = output_dir
        self.replay_mode = replay_mode
        self.game_info = {}
        
        if not replay_mode and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def capture_step(self, agent_index, state, action, result_state=None):
        """Captura un paso completo del juego con representación de mapa"""
        if self.replay_mode:
            return
        
        # Obtener dimensiones del tablero
        walls = state.getWalls()
        width, height = walls.width, walls.height
        
        # Crear una matriz vacía llena de espacios
        game_map = [[' ' for _ in range(height)] for _ in range(width)]
        
        # Agregar paredes (%)
        for x in range(width):
            for y in range(height):
                if walls[x][y]:
                    game_map[x][y] = '%'
        
        # Agregar comida (.)
        food = state.getFood()
        for x in range(width):
            for y in range(height):
                if food[x][y]:
                    game_map[x][y] = '.'
        
        # Agregar cápsulas (o)
        for x, y in state.getCapsules():
            game_map[x][y] = 'o'
        
        # Agregar fantasmas (G)
        for ghost_state in state.getGhostStates():
            ghost_x, ghost_y = int(ghost_state.getPosition()[0]), int(ghost_state.getPosition()[1])
            game_map[ghost_x][ghost_y] = 'G'
        
        # Agregar Pacman (P)
        pacman_x, pacman_y = state.getPacmanPosition()
        game_map[int(pacman_x)][int(pacman_y)] = 'P'
        
        # Convertir el mapa a formato numérico
        # 0: pared (%)
        # 1: espacio vacío ( )
        # 2: comida (.)
        # 3: cápsula (o)
        # 4: fantasma (G)
        # 5: Pacman (P)
        numeric_map = []
        for x in range(width):
            row = []
            for y in range(height):
                if game_map[x][y] == '%':
                    row.append(0)
                elif game_map[x][y] == ' ':
                    row.append(1)
                elif game_map[x][y] == '.':
                    row.append(2)
                elif game_map[x][y] == 'o':
                    row.append(3)
                elif game_map[x][y] == 'G':
                    row.append(4)
                elif game_map[x][y] == 'P':
                    row.append(5)
                else:
                    row.append(1)  # Default a espacio vacío
            numeric_map.append(row)
        
        # Datos del paso
        step_data = {
            'timestamp': datetime.now().isoformat(),
            'agent_index': agent_index,
            'action': action,
            'score': state.getScore(),
            'is_win': state.isWin(),
            'is_lose': state.isLose(),
            'game_over': state.isWin() or state.isLose(),
            # Mapa como matriz numérica (formato JSON)
            'map_matrix': json.dumps(numeric_map)
        }
        
        self.current_game_data.append(step_data)
    
    def set_game_info(self, layout_name, seed):
        """Guarda información del juego"""
        self.game_info = {
            'layout': layout_name,
            'seed': seed,
            'timestamp': datetime.now().isoformat()
        }
    
    def save_game_data(self, game_id):
        """Guarda los datos del juego actual"""
        if self.replay_mode:
            return
        f = lambda x: int(x[5:-4])
        
        if not os.path.exists(self.output_dir):
            game_id = 0
        else:
            # si hay juegos guardados, el id es el siguiente
            # game_id = len(os.listdir(self.output_dir))
            game_id = f(sorted(os.listdir(self.output_dir), key=f)[-1])
        # Crear el nombre del archivo
        # del timestamp solo nos quedamos con la fecha dia/mes/año
        steps_filename = os.path.join(self.output_dir, f"game_{game_id}.csv")
        
        # Guardar los pasos del juego
        fieldnames = [
            'timestamp', 'agent_index', 'action', 'score',
            'is_win', 'is_lose', 'game_over', 'map_matrix'
        ]
        
        with open(steps_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for step in self.current_game_data:
                writer.writerow(step)
        
        print(f"Datos del juego {game_id} guardados en {steps_filename}")
        self.current_game_data = []
    
    def _visualize_map(self, numeric_map):
        """Convierte la matriz numérica a un mapa visual (para debug)"""
        visual_map = []
        for row in numeric_map:
            visual_row = []
            for cell in row:
                if cell == 0:
                    visual_row.append('%')  # Pared
                elif cell == 1:
                    visual_row.append(' ')  # Espacio vacío
                elif cell == 2:
                    visual_row.append('.')  # Comida
                elif cell == 3:
                    visual_row.append('o')  # Cápsula
                elif cell == 4:
                    visual_row.append('G')  # Fantasma
                elif cell == 5:
                    visual_row.append('P')  # Pacman
                else:
                    visual_row.append('?')  # Desconocido
            visual_map.append(''.join(visual_row))
        return '\n'.join(visual_map)
# import csv
# import json
# from datetime import datetime
# import os

# class GameDataCollector:
#     def __init__(self, output_dir="pacman_data",replay_mode=False):
#         self.current_game_data = []  # Datos del juego actual
#         self.output_dir = output_dir
#         self.replay_mode = replay_mode
#         # Crear directorio si no existe
#         if not os.path.exists(output_dir):
#             os.makedirs(output_dir)
    
#     def capture_step(self, agent_index, state, action, result_state=None):
#         """Captura un paso completo del juego"""
#         if self.replay_mode:
#             # Si estamos en modo replay, no capturamos datos
#             return
#         step_data   = {
#             'timestamp': datetime.now().isoformat(),
#             'agent_index': agent_index,
#             'action': action,
#             # Estado actual
#             'pacman_x': state.getPacmanPosition()[0],
#             'pacman_y': state.getPacmanPosition()[1],
#             'score': state.getScore(),
#             'food_count': state.getNumFood(),
#             # Estado del mapa (serializado como string)
#             'food_matrix': self._serialize_grid(state.getFood()),
#             'wall_matrix': self._serialize_grid(state.getWalls()),
#             # Posiciones de fantasmas (serializado como string)
#             'ghost_positions': json.dumps(state.getGhostPositions()),
#             'capsules': json.dumps(state.getCapsules()),
#             # Estado del juego
#             'is_win': state.isWin(),
#             'is_lose': state.isLose(),
#             'game_over': state.isWin() or state.isLose()
#         }
        
#         # Si hay un estado resultado, agregar información adicional
#         if result_state:
#             step_data['score_change'] = result_state.getScore() - state.getScore()
#             step_data['food_eaten'] = state.getNumFood() - result_state.getNumFood()
        
#         self.current_game_data.append(step_data)
    
#     def _serialize_grid(self, grid):
#         """Convierte una Grid a string para guardar en CSV"""
#         return ','.join([''.join(['1' if grid[x][y] else '0' for y in range(grid.height)]) 
#                         for x in range(grid.width)])
    
#     def save_game_data(self, game_id):
#         if self.replay_mode:
#             return  # No guardar en modo reproducción
#         # El id viene dado por los juegos que ya hay guardados en el directorio
#         # si no hay juegos guardados, el id es 0
#         if not os.path.exists(self.output_dir):
#             game_id = 0
#         else:
#             # si hay juegos guardados, el id es el siguiente
#             game_id = len(os.listdir(self.output_dir))
#         # Crear el nombre del archivo
#         # del timestamp solo nos quedamos con la fecha dia/mes/año
#         filename = os.path.join(self.output_dir, f"game_{game_id}.csv")
        
#         # Definir las columnas del CSV
#         fieldnames = [
#             'timestamp', 'agent_index', 'action', 
#             'pacman_x', 'pacman_y', 'score', 'food_count',
#             'food_matrix', 'wall_matrix', 'ghost_positions', 'capsules',
#             'is_win', 'is_lose', 'game_over', 'score_change', 'food_eaten'
#         ]
        
#         with open(filename, 'w', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
            
#             for step in self.current_game_data:
#                 # Asegurarse de que todas las columnas estén presentes
#                 row = {field: step.get(field, '') for field in fieldnames}
#                 writer.writerow(row)
        
#         print(f"Datos del juego {game_id} guardados en {filename}")
        
#         # Reiniciar para el siguiente juego
#         self.current_game_data = []
    
#     def save_summary_csv(self):
#         """Guarda un resumen de todos los juegos en un CSV"""
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
#         filename = os.path.join(self.output_dir, f"games_summary_{timestamp}.csv")
        
#         # Esta función podría implementarse para crear un resumen
#         pass

