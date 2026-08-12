import os
from datetime import datetime

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from Metricas import calcular_metricas


class Prediccion:
    def __init__(self, data_dir, model_path, config_path, verbose=True):
        self.verbose = verbose
        self.X_test, self.y_test = self.load_test_data(data_dir=data_dir)
        self.model = self.load_model(model_path=model_path, config_path=config_path)
        
    def load_test_data(self, data_dir='experiment_results/datasets'):
        """Carga el conjunto de test"""
        test_path = os.path.join(data_dir, 'datasets/test_data.pt')
        
        if not os.path.exists(test_path):
            raise FileNotFoundError(
                f"No se encontró el archivo de test: {test_path}\n"
                "Asegúrate de entrenar primero (main.py con --train)."
            )
        
        test_data = torch.load(test_path)
        X_test = test_data['X']
        y_test = test_data['y']
        
        if self.verbose:
            print(f"✅ Conjunto de test cargado: {len(X_test)} muestras")
        
        return X_test, y_test


    def load_model(self, model_path='experiment_results/models/best_model.pth', 
                config_path='experiment_results/model_config.json'):
        """Carga el modelo entrenado con su configuración"""
        import json
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"No se encontró el modelo: {model_path}\n"
                "Asegúrate de entrenar primero (main.py con --train)."
            )

        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No se encontró la configuración: {config_path}\n"
                "Asegúrate de entrenar primero (main.py con --train)."
            )
        
        # Cargar configuración (se conserva en self.config: las subclases leen de
        # ella parámetros del modelo, como el rango de pre-activaciones)
        with open(config_path, 'r') as f:
            config = json.load(f)
        self.config = config

        if self.verbose:
            print(f"   Arquitectura: {config['architecture']}")
            print(f"   Mejor semilla usada: {config['best_seed']}")
        
        # Crear modelo con la arquitectura correcta
        from Trainer import ConfigurableNN
        
        model = ConfigurableNN(
            input_size=config['input_size'],
            hidden_layers=config['hidden_layers'],
            num_classes=config['num_classes']
        )
        
        # Cargar pesos
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        if self.verbose:
            print(f"✅ Modelo cargado desde: {model_path}")
        
        return model
    
    def calculate_metrics(self, y_pred, y_proba=None, num_clases=None):
        """
        Calcula el conjunto completo de métricas de clasificación delegando en
        Metricas.calcular_metricas. Devuelve un dict con tasas en fracción
        (0-1), MCC/kappa en su rango natural, matriz de confusión, métricas por
        clase y -si se pasan probabilidades- ROC-AUC y log-loss.

        Mantiene las claves clásicas (accuracy, conf_matrix, sensitivity,
        specificity, ppv, npv, tp/tn/fp/fn) para que save_results_txt y el
        resto del código sigan funcionando sin cambios.
        """
        y_true = self.y_test.cpu().numpy() if hasattr(self.y_test, 'cpu') else np.asarray(self.y_test)
        if num_clases is None:
            num_clases = getattr(self.model, 'num_classes', None)
        return calcular_metricas(y_true, y_pred, y_proba=y_proba, num_clases=num_clases)


    def save_results_txt(self, metrics, inference_time, n_samples, save_path,
                         titulo="RESULTADOS DE PREDICCIÓN PLANA (SIN CIFRADO)",
                         descripcion="Predicción estándar sin cifrado homomórfico"):
        """Guarda resultados en archivo de texto (Soporta Binario y Multiclase)"""

        with open(save_path, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write(titulo + "\n")
            f.write("=" * 70 + "\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Tipo: {descripcion}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("⏱️  RENDIMIENTO\n")
            f.write("─" * 70 + "\n")
            f.write(f"Tiempo total de inferencia: {inference_time:.4f} segundos\n")
            f.write(f"Muestras procesadas: {n_samples}\n")
            f.write(f"Tiempo promedio por muestra: {inference_time/n_samples*1000:.2f} ms\n")
            f.write(f"Throughput: {n_samples/inference_time:.2f} muestras/segundo\n\n")
            
            f.write("📊 PRECISIÓN GLOBAL\n")
            f.write("─" * 70 + "\n")
            f.write(f"Accuracy: {metrics['accuracy']*100:.2f}%\n")
            f.write(f"Predicciones correctas: {int(round(metrics['accuracy']*n_samples))}/{n_samples}\n")
            f.write(f"Predicciones incorrectas: {n_samples - int(round(metrics['accuracy']*n_samples))}/{n_samples}\n\n")
            
            f.write("📈 MATRIZ DE CONFUSIÓN\n")
            f.write("─" * 70 + "\n")
            cm = metrics['conf_matrix']
            
            if cm.size == 4:
                # --- FORMATO BINARIO (Mantiene etiquetas originales) ---
                f.write("                 Predicho\n")
                f.write("                 Negativo  Positivo\n")
                f.write(f"Real  Negativo     {cm[0][0]:4d}      {cm[0][1]:4d}\n")
                f.write(f"      Positivo     {cm[1][0]:4d}      {cm[1][1]:4d}\n\n")
                
                f.write("🔬 MÉTRICAS CLÍNICAS (Binarias)\n")
                f.write("─" * 70 + "\n")
                f.write(f"Sensibilidad (Recall Pos):         {metrics['sensitivity']*100:.2f}%\n")
                f.write(f"Especificidad (Recall Neg):        {metrics['specificity']*100:.2f}%\n")
                f.write(f"Valor Predictivo Positivo (PPV):   {metrics['ppv']*100:.2f}%\n")
                f.write(f"Valor Predictivo Negativo (NPV):   {metrics['npv']*100:.2f}%\n\n")
                
                f.write(f"Verdaderos Positivos (TP):  {metrics['tp']}\n")
                f.write(f"Verdaderos Negativos (TN):  {metrics['tn']}\n")
                f.write(f"Falsos Positivos (FP):      {metrics['fp']}\n")
                f.write(f"Falsos Negativos (FN):      {metrics['fn']}\n\n")
            else:
                # --- FORMATO MULTICLASE (Generico para N clases) ---
                num_classes = cm.shape[0]
                f.write(f"Matriz de confusión para {num_classes} clases:\n\n")
                header = "      " + "".join([f"Cl{i:<5}" for i in range(num_classes)])
                f.write(header + "\n")
                for i, row in enumerate(cm):
                    row_str = f"Cl{i:<3} " + "".join([f"{val:<6}" for val in row])
                    f.write(row_str + "\n")
                f.write("\n")
                
                f.write("🔬 MÉTRICAS MULTICLASE (Macro-Averaged)\n")
                f.write("─" * 70 + "\n")
                f.write(f"Recall (Promedio):                {metrics['sensitivity']*100:.2f}%\n")
                f.write(f"Precisión (PPV Promedio):         {metrics['ppv']*100:.2f}%\n")
                f.write(f"NPV (Referencia Clase 0):         {metrics['npv']*100:.2f}%\n")
                f.write(f"Nota: TN, TP, FP, FN no se desglosan en modo multiclase.\n\n")
            
            f.write("=" * 70 + "\n")
            f.write("Archivo generado por Prediccion.save_results_txt\n")
            f.write("=" * 70 + "\n")
            
        if self.verbose:
            print(f"  📄 Resultados guardados: {save_path}")


    def plot_confusion_matrix(self, conf_matrix, save_path, titulo='Matriz de Confusión'):
        """Genera y guarda la imagen de la matriz de confusión (binaria o multiclase)."""
        plt.figure(figsize=(10, 8))

        # Etiquetas según el número de clases
        if conf_matrix.shape[0] == 2:
            labels = ['Negativo', 'Positivo']
        else:
            labels = [f'Clase {i}' for i in range(conf_matrix.shape[0])]

        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)

        ax = plt.gca()
        ax.set_xlabel('Clase Predicha')
        ax.set_ylabel('Clase Real')
        ax.set_title(titulo)

        # Fijar ticks explícitamente para evitar el warning de FixedLocator
        ax.set_xticks(np.arange(len(labels)) + 0.5)
        ax.set_yticks(np.arange(len(labels)) + 0.5)
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close() # Importante cerrar la figura para liberar memoria
