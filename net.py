import os
import glob
import json
import csv
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from pacman_types import Seed
# Fijamos todas las semillas para reproducibilidad
# torch.manual_seed(42)
# random.seed(42)
torch.manual_seed(Seed.get_value())
random.seed(Seed.get_value())
# Constantes
INPUT_SIZE = None  # Se determinará en tiempo de ejecución basado en el tamaño del mapa
HIDDEN_SIZE = 128
NUM_ACTIONS = 5  # Stop, North, South, East, West
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 100
MODELS_DIR = "models"

# Mapeo de acciones a índices
ACTION_TO_IDX = {
    'Stop': 0,
    'North': 1,
    'South': 2, 
    'East': 3,
    'West': 4
}

# Mapeo de índices a acciones
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
# Esto es obligatorio para poder usar dataloaders en pytorch
class PacmanDataset(Dataset):
    def __init__(self, maps, actions):
        self.maps = maps
        self.actions = actions
    
    def __len__(self):
        return len(self.maps)
    
    def __getitem__(self, idx):
        map_tensor = torch.FloatTensor(self.maps[idx])
        action_tensor = torch.LongTensor([self.actions[idx]])
        return map_tensor, action_tensor.squeeze()

class PacmanNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PacmanNet, self).__init__()
        
        # Calcular el tamaño total de entrada (aplanar la matriz)
        self.input_features = input_size[0] * input_size[1]
        
        # Capas fully connected (feedforward)
        self.fc1 = nn.Linear(self.input_features, hidden_size * 2)
        self.fc2 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        
        # Activaciones
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # Input shape: (batch_size, height, width)
        #print(f"Forma de entrada: {x.shape}")
        # Aplanar la entrada
        x = x.view(x.size(0), -1)  # Shape: (batch_size, height*width)
        
        # Capas fully connected
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class PacmanEval(nn.Module):
    def __init__(self, input_size:tuple[int, int]):
        super().__init__()

        # y|x dims
        # Defining the items 
        self.input_size = input_size
        self.l1 = nn.Linear(in_features = torch.prod(torch.tensor(input_size)), out_features = 256)
        self.l2 = nn.Linear(in_features = 256, out_features = 128)
        self.l3 = nn.Linear(in_features = 128, out_features = 64)
        self.l4 = nn.Linear(in_features = 64, out_features = 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)


    def forward(self, x:torch.Tensor):
        # Aplanar la entrada
        x = x.view(x.size(0), -1)  # Shape: (batch_size, height*width)
        
        # Capas fully connected
        x = self.relu(self.l1(x))
        x = self.dropout(x)
        x = self.relu(self.l2(x))
        x = self.dropout(x)
        x = self.relu(self.l3(x))
        x = self.dropout(x)
        x = self.l4(x)
        
        return x


def load_and_merge_data(data_dir="pacman_data")->tuple[list[list[list[int]]], list[float]]:
    """Carga todos los archivos CSV de partidas y los combina en un único DataFrame"""
    all_maps = []
    all_evaluations = []
    
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    print(f"Archivos CSV encontrados: {csv_files}")
    
    if not csv_files:
        raise ValueError(f"No se encontraron archivos CSV en {data_dir}")
    
    print(f"Cargando {len(csv_files)} archivos de partidas...")
    
    for csv_file in csv_files:
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                """ # Solo usar movimientos de Pacman (agente 0)
                    if int(row.get('agent_index', 0)) == 0:
                """
                #action = row.get('action')
                eval = row.get('evaluation')
                map_matrix = json.loads(row.get('map_matrix', '[]'))
                
                # Verificar que los datos sean válidos
                if map_matrix:
                    all_maps.append(map_matrix)
                    all_evaluations.append(float(eval))
    
    print(f"Datos cargados: {len(all_maps)} ejemplos")
    return all_maps, all_evaluations

def preprocess_maps(maps):
    """Preprocesa las matrices del juego para preparar los datos de entrada para la red"""
    # Determinar las dimensiones del mapa
    height = len(maps[0])
    width = len(maps[0][0])
    
    # Convertir a numpy array
    processed_maps = np.array(maps).astype(np.float32)
    
    # Normalizar los valores: dividir por 5 (el valor máximo) para obtener valores entre 0 y 1
    processed_maps = processed_maps / 5.0
    
    print(f"Forma de los datos de entrada: {processed_maps.shape}")
    print(f"Tamaño del mapa: {height}x{width}")
    
    return processed_maps, (height, width)


def train_model(model, train_loader, test_loader, device, num_epochs=NUM_EPOCHS):
    """Entrena el modelo con el dataset proporcionado"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    best_accuracy = 0.0
    best_model_state = None
    
    print(f"Comenzando entrenamiento por {num_epochs} épocas...")
    
    for epoch in range(num_epochs):
        # Entrenamiento
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (maps, evals) in enumerate(train_loader):
            maps, evals = maps.to(device), evals.to(device)
            
            # Forward pass
            outputs = model(maps)
            loss = criterion(outputs, evals)
            
            # Backward pass y optimización
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Estadísticas
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += evals.size(0)
            train_correct += predicted.eq(evals).sum().item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(train_loader)}, Loss: {train_loss/(batch_idx+1):.4f}, Acc: {100.*train_correct/train_total:.2f}%')
        
        # Evaluación
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for maps, evals in test_loader:
                maps, evals = maps.to(device), evals.to(device)
                outputs = model(maps)
                loss = criterion(outputs, evals)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += evals.size(0)
                test_correct += predicted.eq(evals).sum().item()
        
        test_accuracy = 100. * test_correct / test_total
        print(f'Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Test Loss: {test_loss/len(test_loader):.4f}, Test Acc: {test_accuracy:.2f}%')
        
        # Guardar el mejor modelo
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model_state = model.state_dict().copy()
            print(f'¡Nuevo mejor modelo con {best_accuracy:.2f}% de precisión!')
    
    # Cargar el mejor modelo
    if best_model_state:
        model.load_state_dict(best_model_state)
        print(f'Modelo final: precisión en test {best_accuracy:.2f}%')
    
    return model

def save_model(model, input_size, model_path="models/pacman_eval_model.pth"):
    """Guarda el modelo entrenado"""
    if not os.path.exists(os.path.dirname(model_path)):
        os.makedirs(os.path.dirname(model_path))
    
    # Guardar el modelo junto con información sobre el tamaño de entrada
    model_info = {
        'model_state_dict': model.state_dict(),
        'input_size': input_size,
    }
    torch.save(model_info, model_path)
    print(f'Modelo guardado en {model_path}')

def main():
    import time
    start_time = time.time()
    # Verificar disponibilidad de GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar datos--ok
    maps, evals = load_and_merge_data(data_dir="train")
    
    # Preprocesar mapas-- ok
    maps, input_size = preprocess_maps(maps)
    
    import code; code.interact(local=locals())
    # Dividir en conjunto de entrenamiento y test--ok
    X_train, X_test, y_train, y_test = train_test_split(
        maps, evals, test_size=0.2, random_state=42, stratify=evals
    )
    
    # Crear datasets--ok
    train_dataset = PacmanDataset(X_train, y_train)
    test_dataset = PacmanDataset(X_test, y_test)
    
    # Crear dataloaders--ok
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Crear modelo
    model = PacmanEval(input_size).to(device)
    print(f"Modelo creado: {model}")
    
    # Entrenar modelo
    trained_model = train_model(model, train_loader, test_loader, device)
    
    # Guardar modelo
    save_model(trained_model, input_size)
    print(f"Tiempo total de ejecución: {time.time() - start_time:.2f} segundos")
if __name__ == "__main__":
    main()
