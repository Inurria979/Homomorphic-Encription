"""
RED NEURONAL Y ENTRENAMIENTO
============================
- ConfigurableNN: red densa con arquitectura configurable (entrada -> capas
  ocultas -> salida), entrenada con ReLU de PyTorch. En predicción homomórfica
  la ReLU se sustituye por un polinomio ajustado a su rango real (ver
  PrediccionHomomorfica).
- Trainer: entrenamiento con early stopping. NO usa el conjunto de test durante
  el entrenamiento/validación (se reserva para la evaluación final).
Sirve para cualquiera de los datasets del proyecto, no solo Breast Cancer.
"""

import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset

class ConfigurableNN(nn.Module):
    """
    Red neuronal con arquitectura configurable
    Entrenada con ReLU de PyTorch; en cifrado la ReLU se aproxima por un polinomio
    """
    def __init__(self, input_size=30, hidden_layers=[16, 8], num_classes=2):
        super(ConfigurableNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.num_classes = num_classes
        self.architecture = [input_size] + hidden_layers + [num_classes]
        # 1. INICIALIZAR el ModuleList PRIMERO
        self.layers = nn.ModuleList() 
        prev_size = input_size
        
        # 2. Construir las capas ocultas
        if hidden_layers:
            for h_size in hidden_layers:
                self.layers.append(nn.Linear(prev_size, h_size))
                prev_size = h_size
                
        # 3. La última capa (la que conecta al output)
        self.classifier = nn.Linear(prev_size, num_classes)
        
        self.activation = nn.ReLU()
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Inicializa pesos con Xavier"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass con ReLU. La capa final va sin activación porque
        CrossEntropyLoss aplica el softmax internamente."""
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.classifier(x)
        return x


    def get_config(self):
        """Devuelve la configuración del modelo"""
        arch = f'{self.input_size} -> {self.num_classes}'
        if self.hidden_layers:
            arch = f"{self.input_size} -> {' -> '.join(map(str, self.hidden_layers))} -> {self.num_classes}"
        return {
            'input_size': self.input_size,
            'hidden_layers': self.hidden_layers,
            'num_classes': self.num_classes,
            'architecture': arch
        }


class Trainer:

    def __init__(self, model, train_loader, val_loader, test_loader, scaler, learning_rate=0.001, rg=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.scaler = scaler
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Mover el modelo al dispositivo ANTES de crear el optimizador, para que
        # el optimizador tome como referencia los parámetros ya ubicados
        self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        # weight_decay = regularización L2 (parámetro rg)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=rg)
        
        # Historial de métricas
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train_epoch(self):
        """Entrena una época"""
        self.model.train()
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in self.train_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
            
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        return epoch_loss / len(self.train_loader), 100 * correct / total
    
    def validate(self):
        """Valida el modelo"""
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        return val_loss / len(self.val_loader), 100 * correct / total
    
    def train(self, epochs=100, early_stopping_patience=15, verbose=True):
        """Entrena el modelo con early stopping"""
        best_val_loss = float('inf')
        best_val_acc = 0
        patience_counter = 0
        best_state = None

        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"      Época {epoch+1}/{epochs} - Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")

            # Early stopping
            # deepcopy: state_dict() devuelve referencias a los tensores vivos,
            # una copia superficial quedaría mutada por el propio entrenamiento
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"      Early stopping en época {epoch+1}")
                    break

        # Cargar mejor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)

        # Devolvemos la accuracy de la época con mejor val_loss (la del estado
        # restaurado), no la de la última época
        return best_val_loss, best_val_acc

    def last_train(self, epochs=100, early_stopping_patience=15, verbose=True):
        """
        Reentrena el mejor modelo usando train + validación juntos (ya no se
        reserva validación: el conjunto de test sigue intacto). Early stopping
        sobre la pérdida de entrenamiento.
        """
        best_train_loss = float('inf')
        patience_counter = 0
        best_state = None

        # 1. Accedemos a los Datasets originales dentro de tus loaders
        full_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset])
        
        # 2. Creamos el cargador definitivo con TODOS los datos
        self.train_loader = DataLoader(
            full_dataset, 
            batch_size=self.train_loader.batch_size, 
            shuffle=True
        )
        
        for epoch in range(1, epochs+1):
            train_loss, train_acc = self.train_epoch()
            if verbose:
                if epoch % 10 == 0:
                    print(f'Iteración {epoch} del entrenamiento final')
                    print(f"Pérdida de entrenamiento {train_loss}")
                    print(f"Acc de entrenamiendo {train_acc}")
            
            # Early stopping
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                best_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"      Early stopping en época {epoch+1}")
                    break
        # Cargar mejor modelo
        if best_state is not None:
            self.model.load_state_dict(best_state)
