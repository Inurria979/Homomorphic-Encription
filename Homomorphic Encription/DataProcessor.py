"""
PROCESAMIENTO DE DATOS
======================
Descarga datasets de UCI, los limpia (codificación de categóricas, imputación
de NaN), hace el split estratificado train/val/test y prepara los DataLoaders
(escalado y PCA ajustados SOLO con el conjunto de entrenamiento, para no filtrar
información del test).

IDs de dataset soportados:
    17  -> Breast Cancer Wisconsin
    891 -> CDC Diabetes Health Indicators
    544 -> Obesity Levels (hábitos alimenticios y condición física)
    27  -> Credit Approval
"""

import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from ucimlrepo import DatasetNotFoundError

from ResultadosStore import DATASET_NOMBRES


class DataProcessor:

    def __init__(self, id=17):
        self.dataset_id = id
        self.dataset_nombre = DATASET_NOMBRES.get(id, f'UCI id {id}')
        self.X, self.y = self.load_data_from_uci(id=id)
        self.len_x_test = 0
        self.Scaler = StandardScaler()

    def load_data_from_uci(self, id=17):
        """Carga dataset de UCI"""
        try:
            from ucimlrepo import fetch_ucirepo
            dataset = fetch_ucirepo(id=id)

            # .copy() para modificar sin avisos de vista de pandas (SettingWithCopy)
            X_df = dataset.data.features.copy()
            y = dataset.data.targets.values.ravel()

            # Cap de tamaño: los datasets muy grandes (Diabetes) dispararían el
            # tiempo de la predicción homomórfica a horas. Muestreo estratificado.
            max_samples = 20000
            if len(X_df) > max_samples:
                print(f"[*] Dataset muy grande ({len(X_df)}). Reduciendo a {max_samples} para agilidad.")
                X_df, _, y, _ = train_test_split(X_df, y, train_size=max_samples, stratify=y, random_state=42)

            # Codificar columnas categóricas a numéricas
            for col in X_df.columns:
                if X_df[col].dtype == 'object' or isinstance(X_df[col].iloc[0], str):
                    # Rellenamos NaN temporalmente para que LabelEncoder no falle;
                    # el Imputer se encarga después de los NaN numéricos reales
                    X_df[col] = X_df[col].fillna('unknown')
                    X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

            # Imputar los NaN numéricos con la media de cada columna
            X = SimpleImputer(strategy='mean').fit_transform(X_df.values)

            # Codificar las etiquetas a enteros 0..n_clases-1
            y = LabelEncoder().fit_transform(y)

            return X, y
        except DatasetNotFoundError:
            print(f"\n[!] El ID {id} no está disponible para descarga directa en UCI.")
            print("Prueba con ID 225 (Diabetes), ID 45 (Heart) o ID 174 (Parkinsons).")
            sys.exit(-1)
        except ImportError:
            print("⚠️  ucimlrepo no instalado. Generando datos sintéticos...")
            return self._generate_synthetic_data()
    
    def _generate_synthetic_data(self):
        """Genera datos sintéticos"""
        np.random.seed(42)
        n_samples = 569
        X = np.random.randn(n_samples, 30)
        X[:285, :10] += 1.5
        X[285:, :10] -= 1.5
        y = np.array([1] * 285 + [0] * 284)
        return X, y

    def show_dataset(self, x_f, y_f):
        df = pd.concat([pd.DataFrame(x_f), pd.DataFrame(y_f, columns=['target'])], axis=1)
        return df


    def cov_matrix(self, X_f, y_f):
        df_X = pd.DataFrame(X_f)
        df_y = pd.Series(y_f).to_frame()
        df = pd.concat([df_X, df_y], axis=1)

        # Estandarizar
        
        df_std = pd.DataFrame(self.Scaler.fit_transform(df))

        cov_m = df_std.cov()
        
        # Pintar
        plt.figure(figsize=(12, 10))
        sns.heatmap(cov_m, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title("Matriz de Covarianza (Datos Estandarizados)")
        plt.show()

    def principal_component_analysis(self, X_f, pca_num_features = 5):

        self.Scaler.fit(X_f)
        X_scaled = self.Scaler.transform(X_f)

        self.pca = PCA(n_components=pca_num_features)
        X_pca = self.pca.fit_transform(X_scaled)

        return X_pca
    
    def split_data(self, test_size=0.2, val_size=0.1, seed=42):
        """
        Split estratificado en train/val/test. El conjunto de test se fija con
        una semilla maestra constante para que sea EL MISMO en todas las
        semillas del Monte Carlo (solo se remezclan train y val con `seed`).
        """
        master_seed = 42
        # 1. Separar el Test set del resto (siempre igual, semilla fija)
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=master_seed,
            stratify=self.y
        )

        # 2. Separar el resto en Train y Validation.
        # val_size está sobre el total, así que ajustamos su tamaño relativo
        val_rel_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_rel_size, 
            random_state=seed, 
            stratify=y_train_val
        )
        self.len_x_test = len(X_test)
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)



    def transform_and_load(self, train_set, val_set, test_set, pca_features=0, batch_size=32):
        """
        Ajusta Scaler/PCA, transforma los datos y crea DataLoaders.
        """
        X_train, y_train = train_set
        X_val, y_val = val_set
        X_test, y_test = test_set

        # 1. Ajustar Scaler SOLO con entrenamiento y transformar todos
        self.Scaler.fit(X_train)
        X_train_scaled = self.Scaler.transform(X_train)
        X_val_scaled = self.Scaler.transform(X_val)
        X_test_scaled = self.Scaler.transform(X_test)

        # 2. Si hay PCA, ajustar SOLO con entrenamiento escalado
        if pca_features > 0:
            self.pca = PCA(n_components=pca_features)
            X_train_final = self.pca.fit_transform(X_train_scaled)
            X_val_final = self.pca.transform(X_val_scaled)
            X_test_final = self.pca.transform(X_test_scaled)
        else:
            X_train_final = X_train_scaled
            X_val_final = X_val_scaled
            X_test_final = X_test_scaled
        
        # 3. Convertir a tensores
        train_ds = TensorDataset(torch.FloatTensor(X_train_final), torch.LongTensor(y_train))
        val_ds = TensorDataset(torch.FloatTensor(X_val_final), torch.LongTensor(y_val))
        test_ds = TensorDataset(torch.FloatTensor(X_test_final), torch.LongTensor(y_test))

        # 4. Crear DataLoaders
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader, test_loader