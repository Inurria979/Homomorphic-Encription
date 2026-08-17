"""
RED NEURONAL Y ENTRENAMIENTO
============================
- ConfigurableNN: red densa con arquitectura configurable (entrada -> capas
  ocultas -> salida), entrenada con ReLU de PyTorch. En predicción homomórfica
  la ReLU se sustituye por un polinomio ajustado a su rango real (ver
  PrediccionHomomorfica).
- Trainer: entrenamiento con early stopping. NO usa el conjunto de test durante
  el entrenamiento/validación (se reserva para la evaluación final). Al terminar
  el reentrenamiento mide el rango sobre el que hay que aproximar la ReLU en el
  cifrado, que se publica junto a los pesos.
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

    def __init__(self, model, train_loader, val_loader, test_loader, learning_rate=0.001, rg=0):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Mover el modelo al dispositivo ANTES de crear el optimizador, para que
        # el optimizador tome como referencia los parámetros ya ubicados
        self.model.to(self.device)

        # CrossEntropyLoss = LogSoftmax + NLLLoss (espera logits crudos)
        self.criterion = nn.CrossEntropyLoss()
        # weight_decay = regularización L2 (parámetro rg)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=rg)
        
        # Rango de pre-activaciones medido al final de last_train
        self.rango_preactivaciones = None

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
        best_epoch = 0
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
                # Época (1-indexada) en la que la validación fue óptima: es el
                # hiperparámetro que el reentrenamiento final reutiliza
                best_epoch = epoch + 1
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
        # restaurado), no la de la última época, junto con esa época
        return best_val_loss, best_val_acc, best_epoch

    def medir_rango_preactivaciones(self):
        """
        Mide el rango [a, b] de las pre-activaciones (las entradas a cada ReLU)
        propagando en claro los datos de entrenamiento por el modelo.

        Es el rango sobre el que hay que aproximar la ReLU en el cifrado. NO es
        [-1, 1]: aunque los pesos rondan [-1, 1], la pre-activación z = Σ wᵢxᵢ + b
        suma sobre 12-30 entradas (fan-in) y las features van estandarizadas
        (~[-6, 8]), así que z alcanza ±8…±15 en las redes del proyecto.

        La capa de salida (classifier) no lleva activación, por eso solo se
        instrumentan las capas ocultas del modelo. Devuelve None si no hay capas
        ocultas: sin activación no hay nada que aproximar.
        """
        if not self.model.layers:
            return None

        minimos, maximos = [], []

        def hook(_modulo, _entrada, salida):
            # Sin valor de retorno: lo que devuelve un forward hook sustituiría a
            # la salida de la capa
            minimos.append(float(salida.min()))
            maximos.append(float(salida.max()))

        handles = [capa.register_forward_hook(hook) for capa in self.model.layers]
        estaba_entrenando = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for batch_X, _ in self.train_loader:
                    self.model(batch_X.to(self.device))
        finally:
            for h in handles:
                h.remove()
            self.model.train(estaba_entrenando)

        if not minimos:
            return None
        return min(minimos), max(maximos)

    def last_train(self, epochs, verbose=True):
        """
        Reentrena el modelo con train + validación juntos durante un número FIJO
        de épocas (el conjunto de test sigue intacto).

        `epochs` procede del muestreo Monte Carlo: es la mediana de las épocas en
        las que la pérdida de validación fue óptima a lo largo de las semillas.
        No se hace early stopping aquí: al fusionar validación con entrenamiento
        ya no queda ningún conjunto sobre el que medir generalización, y parar por
        la pérdida de entrenamiento no es un criterio válido (esa pérdida casi
        siempre decrece, así que agotaría las épocas disponibles). El modelo final
        es el de la última época; no se restaura ningún estado intermedio.

        Al terminar mide el rango de las pre-activaciones sobre esos mismos datos
        (con los pesos ya definitivos) y lo devuelve: es el intervalo en el que
        PrediccionHomomorfica ajusta el polinomio que sustituye a la ReLU, y viaja
        con el modelo en model_config.json igual que los pesos y los sesgos.
        """
        # 1. Accedemos a los Datasets originales dentro de los loaders
        full_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset])

        # 2. Creamos el cargador definitivo con TODOS los datos
        self.train_loader = DataLoader(
            full_dataset,
            batch_size=self.train_loader.batch_size,
            shuffle=True
        )

        print(f"   Reentrenando con train+val ({len(full_dataset)} muestras) "
              f"durante {epochs} épocas fijas")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            if verbose and epoch % 10 == 0:
                print(f'   Época {epoch}/{epochs} del entrenamiento final - '
                      f'Pérdida {train_loss:.4f}, Acc {train_acc:.2f}%')

        self.rango_preactivaciones = self.medir_rango_preactivaciones()

        if verbose:
            if self.rango_preactivaciones is None:
                print("   Red sin capas ocultas: no hay rango de pre-activaciones")
            else:
                a, b = self.rango_preactivaciones
                print(f"   Rango de pre-activaciones (train+val): [{a:.2f}, {b:.2f}]")

        return self.rango_preactivaciones
