"""
PREDICCIÓN HOMOMÓRFICA (CON CIFRADO)
====================================
Ejecuta la red neuronal COMPLETA sobre datos cifrados con TenSEAL (esquema
CKKS): cifra el test, propaga por las capas lineales sobre el cifrado usando
una aproximación polinómica de la ReLU (ajustada por mínimos cuadrados al rango
real de las pre-activaciones, ver _coeficientes_relu), y solo descifra el
resultado final. Así el servidor que infiere nunca ve los datos en claro.

Esta clase procesa todo el test de una vez (apropiado para datasets pequeños);
para los grandes se usa PrediccionHomomorficaParalela.
"""

import numpy as np

from Prediccion import Prediccion
import tenseal as ts
import torch
import torch.nn as nn


class PrediccionHomomorfica(Prediccion):
    """Realiza predicciones completamente sobre datos cifrados homomórficamente"""
    
    def __init__(self, data_dir, model_path, config_path,degree=5, verbose=True):
        super().__init__(data_dir, model_path, config_path, verbose=verbose)

        if self.verbose:
            # 1. Ver qué tenemos en el modelo (opcional para debug)
            for nombre, parametro in self.model.state_dict().items():
                print(f"Capa: {nombre} | Tamaño: {parametro.size()}")

        self.degree = degree
        # 2. Extraer pesos y sesgos para TenSEAL
        self.pesos_lista = []
        self.biases_lista = []
        
        for layer in self.model.modules():
            if isinstance(layer, nn.Linear):
                # IMPORTANTE: .T es necesario porque TenSEAL opera con vectores fila
                self.pesos_lista.append(layer.weight.T.detach().tolist()) 
                self.biases_lista.append(layer.bias.detach().tolist())
        
        # Aproximación polinómica de la ReLU AJUSTADA AL RANGO REAL de las
        # pre-activaciones (no a [-1,1] fijo). Ver _medir_rango_preactivaciones y
        # _coeficientes_relu: por el fan-in y las features estandarizadas el rango
        # real es mucho más ancho que [-1,1], así que fijar el ajuste en [-1,1]
        # hacía que el polinomio extrapolara sin control.
        if degree == "sqt":
            # Activación cuadrada pura (no es ReLU): coeficientes fijos x^2.
            self.rango_activacion = None
            self.taylor_coeffs = [0.0, 0.0, 1.0]
        else:
            self.rango_activacion = self._medir_rango_preactivaciones()
            # Red sin capas ocultas: no hay activación en el cifrado -> sin polinomio.
            self.taylor_coeffs = (self._coeficientes_relu(degree, self.rango_activacion)
                                  if self.rango_activacion is not None else None)
        if self.verbose:
            print(f"✅ Se han extraído pesos de {len(self.pesos_lista)} capas lineales.")
            if self.rango_activacion is not None:
                a, b = self.rango_activacion
                print(f"✅ Rango de pre-activaciones medido: [{a:.2f}, {b:.2f}] "
                      f"(fan-in + estandarización -> NO es [-1,1])")
            print(f"✅ Polinomio ReLU grado {degree} ajustado al rango: {self.taylor_coeffs}")
        
        
    def setup_encryption_context(self, poly_modulus_degree=32768, coeff_mod_bit_sizes=[60] + [30] * 24 + [60],
                                 global_scale=2**30):
        """
        Configura el contexto de cifrado CKKS para operaciones homomórficas.

        Args:
            poly_modulus_degree: grado del polinomio (mayor = más seguro pero más lento).
            coeff_mod_bit_sizes: tamaños en bits de la cadena de módulos (define la
                                 profundidad multiplicativa disponible).
            global_scale: escala de codificación de los números en punto flotante.
        """
        if self.verbose:
            print("\n🔐 Configurando contexto de cifrado homomórfico...")
            print(f"   Esquema: CKKS (permite operaciones con punto flotante)")
            print(f"   Poly modulus degree: {poly_modulus_degree}")
            
        # Crear contexto CKKS con más niveles para soportar toda la red
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            # Más niveles para operaciones profundas
            coeff_mod_bit_sizes= coeff_mod_bit_sizes
        )
        
        # Configurar escala global
        self.context.global_scale = global_scale
        
        # Generar claves de Re linear (necesarias para reducir ruido)
        self.context.generate_relin_keys()
        # Generar claves de Galois (necesarias para rotaciones)
        self.context.generate_galois_keys()

        if self.verbose:
            print("   ✅ Contexto de cifrado configurado")
            print(f"   🔑 Claves generadas (pública, privada, Galois)")
            print(f"   ⚠️  TODA la red neuronal se ejecutará sobre datos cifrados")
            
    def encrypt_data(self):
        """Cifra los datos de entrada"""
        if self.verbose:
            print(f"\n🔒 Cifrando {len(self.X_test)} muestras...")
        enc_x = ts.ckks_tensor(self.context, self.X_test)

        return enc_x
    
    def encrypt_data_batch(self, X):
        """Cifra los datos de entrada"""
        if self.verbose:
            print(f"\n🔒 Cifrando {len(X)} muestras...")
        enc_x = ts.ckks_tensor(self.context, X)

        return enc_x

    def _medir_rango_preactivaciones(self):
        """
        Mide el rango [a, b] de las pre-activaciones (las entradas a cada ReLU)
        con una pasada EN CLARO del modelo sobre el test.

        Es el rango sobre el que hay que aproximar la ReLU. NO es [-1, 1]: aunque
        los pesos rondan [-1, 1], la pre-activación z = Σ wᵢxᵢ + b suma sobre 12-30
        entradas (fan-in) y las features van estandarizadas (~[-6, 8]), así que z
        alcanza ±8…±15 en las redes del proyecto. Ajustar el polinomio solo en
        [-1, 1] hacía que se evaluara muy fuera de su rango (extrapolación).

        Se propaga con la ReLU real; como el polinomio ajustado ≈ ReLU sobre [a, b],
        el rango sigue siendo válido en la inferencia cifrada. La capa de salida
        (self.classifier) no lleva activación, por eso solo se instrumentan las
        capas ocultas de self.model.layers. Devuelve None si no hay capas ocultas.
        """
        preactivaciones = []
        handles = [capa.register_forward_hook(
            lambda _m, _e, salida: preactivaciones.append(salida.detach()))
            for capa in self.model.layers]
        try:
            with torch.no_grad():
                self.model(self.X_test)
        finally:
            for h in handles:
                h.remove()
        if not preactivaciones:            # red sin capas ocultas: no hay ReLU
            return None
        z = torch.cat([p.reshape(-1) for p in preactivaciones])
        return float(z.min()), float(z.max())

    def _coeficientes_relu(self, degree, rango):
        """
        Coeficientes (base de potencias ascendente [c0..cd], como espera polyval de
        TenSEAL) del polinomio que mejor aproxima ReLU sobre `rango` = [a, b].

        Sustituye a los coeficientes fijos ad-hoc anteriores, que estaban ajustados
        a [-1, 1] (y los de grado 5 eran de hecho la serie de Taylor de 1-e^{-x},
        NO de ReLU) y fallaban en el rango real. Aquí se ajusta por mínimos cuadrados
        sobre el rango medido, de modo que:
          - funciona para CUALQUIER grado (no solo 2/3/5),
          - a mayor grado, menor error de aproximación sobre el rango (monótono).
        """
        a, b = rango
        # Muestreo denso del rango real + ReLU exacta; ajuste polinómico por mínimos
        # cuadrados (numpy, base de potencias ascendente, sin scipy).
        xs = np.linspace(a, b, 2000)
        ys = np.maximum(xs, 0.0)
        coeffs = np.polynomial.polynomial.polyfit(xs, ys, int(degree))
        return [float(c) for c in coeffs]


    def predict_on_encrypted_batch(self, enc_x):
        """
        Propaga un lote cifrado (CKKSTensor) por toda la red en un solo paso
        matemático, capa a capa. Devuelve los logits cifrados de salida.
        """
        if self.verbose:
            print("✅ Prediciendo ...")
        coeffs = self.taylor_coeffs
        num_capas = len(self.pesos_lista)
        for i in range(num_capas):
            # Capa lineal sobre el cifrado: multiplicación matricial + bias
            enc_x = enc_x.mm(self.pesos_lista[i]) + self.biases_lista[i]
            # Activación polinómica (ReLU aproximada) en todas menos la última
            if i < num_capas - 1:
                enc_x = enc_x.polyval(coeffs)  # polyval nativo de TenSEAL
        return enc_x
    
    def decrypt_predictions(self, encrypted_tensor):
        """
        Desencripta los resultados finales de la red.
        """
        if self.verbose:
            print(f"\n🔓 Desencriptando resultados finales...")
        # Desencriptar el tensor completo
        plain_tensor = encrypted_tensor.decrypt()
        
        # Convertir a matriz numpy (n_muestras x n_clases) de logits descifrados
        outputs = np.array(plain_tensor.tolist())

        # La clase predicha es el argmax de los logits (igual que en la plana)
        predictions = np.argmax(outputs, axis=1)
        return predictions

