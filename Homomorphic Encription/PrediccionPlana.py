"""
PREDICCIÓN PLANA (SIN CIFRADO)
==============================
Inferencia estándar del modelo entrenado sobre el conjunto de test, como línea
base con la que comparar la predicción homomórfica. Hereda de Prediccion la
carga de datos/modelo, el cálculo de métricas y el guardado de resultados.
"""

import time

import torch
import torch.nn.functional as F

from Prediccion import Prediccion


class PrediccionPlana(Prediccion):

    def __init__(self, data_dir, model_path, config_path, verbose=True):
        super().__init__(data_dir, model_path, config_path, verbose=verbose)


    def predict_plain(self):
        """Realiza predicciones sin cifrado"""
        if self.verbose:
            print("\n🔮 Realizando predicciones planas (sin cifrado)...")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.model(self.X_test)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        inference_time = time.time() - start_time
        
        if self.verbose:
            print(f"   ✅ Predicciones completadas en {inference_time:.4f} segundos")
            print(f"   ⚡ Tiempo promedio por muestra: {inference_time/len(self.X_test)*1000:.2f} ms")
        
        return predictions.numpy(), probabilities.numpy(), inference_time

