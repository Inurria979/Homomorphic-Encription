"""
PREDICCIÓN HOMOMÓRFICA (CON CIFRADO)
====================================
Ejecuta la red neuronal COMPLETA sobre datos cifrados con TenSEAL (esquema
CKKS): cifra el test, propaga por las capas lineales sobre el cifrado usando
una aproximación polinómica de la ReLU (ajustada por mínimos cuadrados al rango
real de las pre-activaciones, ver _coeficientes_relu), y solo descifra el
resultado final. Así el servidor que infiere nunca ve los datos en claro: el
rango sobre el que se ajusta el polinomio no se mide aquí, viene medido sobre
train+val desde el entrenamiento (Trainer.last_train) y se lee de
model_config.json, igual que el resto de parámetros del modelo.

Los parámetros del esquema (poly_modulus_degree, cadena de módulos y escala) no
hace falta pasárselos: si no se dan, setup_encryption_context los deriva de la
propia red. No son tres decisiones independientes, sino un reparto del mismo
presupuesto de bits que se resuelve en este orden: la PROFUNDIDAD del circuito
fija cuántos primos necesita la cadena; el nivel de SEGURIDAD acota el total de
bits que cabe en cada anillo; dentro de ese margen se busca la mayor escala
posible, que es de lo que depende la PRECISIÓN; y entre las configuraciones que
la alcanzan se escoge la de menor COSTE de inferencia. Pasarlos sigue siendo
posible, y es lo que hace el barrido para comparar configuraciones entre sí.

Esta clase procesa todo el test de una vez (apropiado para datasets pequeños);
para los grandes se usa PrediccionHomomorficaParalela.
"""

import math

import numpy as np

from Prediccion import Prediccion
import tenseal as ts
import torch.nn as nn


class PrediccionHomomorfica(Prediccion):
    """Realiza predicciones completamente sobre datos cifrados homomórficamente"""

    # --- Parámetros del esquema CKKS ---------------------------------------
    # Los tres parámetros del contexto (poly_modulus_degree, coeff_mod_bit_sizes
    # y global_scale) NO se eligen por separado: se reparten un mismo presupuesto
    # de bits. Las constantes de aquí son las que rigen ese reparto, en el orden
    # en que se aplica: profundidad -> seguridad -> precisión -> coste.

    # 2. SEGURIDAD. Cotas del módulo de coeficientes en bits, por tamaño de
    # anillo y nivel de seguridad (Albrecht et al., 2019; es la tabla que
    # implementa SEAL).
    #
    # La seguridad NO consume presupuesto: lo limita. Es una cota superior sobre
    # el total de la cadena, y es lo único que impone. Gastar por debajo del
    # máximo no desperdicia nada, eleva el nivel efectivo por encima de 128 bits,
    # de modo que toda configuración que la biblioteca acepte satisface como
    # mínimo la seguridad estándar.
    MAX_BITS_SEGURIDAD = {
        8192:  {128: 218, 192: 152, 256: 118},
        16384: {128: 438, 192: 305, 256: 237},
        32768: {128: 881, 192: 611, 256: 476},
    }
    # Nivel estándar en cifrado homomórfico y el que TenSEAL aplica por defecto:
    # es la cota que se respeta al repartir el presupuesto. El nivel realmente
    # alcanzado se calcula después (ver seguridad_efectiva) y suele ser mayor,
    # porque la cadena no llega a agotar el máximo disponible.
    SEGURIDAD_MINIMA = 128

    POLY_CANDIDATOS = (8192, 16384, 32768)

    # Primos de los extremos de la cadena: el primero aloja el resultado final y
    # el último es el primo especial de conmutación de claves. Ninguno de los dos
    # aporta profundidad multiplicativa, y por eso se descuentan del presupuesto.
    PRIMO_ESPECIAL = 60

    # 3. PRECISIÓN. Techo de la escala: el primer primo tiene 60 bits y aloja la
    # escala MÁS la parte entera del resultado. Con 50 quedan 10 bits de entero
    # (valores hasta ~1024, de sobra para unos logits); con 58 quedan 2 y el
    # resultado deja de tener sentido (error medido: 4.0).
    ESCALA_MAXIMA = PRIMO_ESPECIAL - 10
    # Escala por debajo de la cual no se acepta un anillo: se prueba el siguiente.
    # Quedarse corto de escala cuesta precisión, y en multiclase el ruido llega a
    # cambiar la clase predicha (medido: con 2^25 y grado 7 el resultado deja de
    # ser reproducible entre corridas).
    ESCALA_OBJETIVO = 40
    # Suelo por tamaño de anillo: SEAL exige primos distintos y ≡ 1 (mod 2N), y
    # con primos muy pequeños no encuentra suficientes candidatos.
    ESCALA_MINIMA_POR_POLY = {8192: 20, 16384: 25, 32768: 26}

    # 4. COSTE. Mayor anillo admisible. Cada duplicación multiplica por ~10 el
    # coste de una multiplicación, así que solo se sube cuando hace falta para
    # alcanzar ESCALA_OBJETIVO. El tope estuvo en 16384 mientras se generaban
    # claves de Galois, que a 32768 agotaban la RAM; ya no se generan (la
    # inferencia usa mm y polyval sobre CKKSTensor, que no rotan), y medido al
    # quitarlas el pico de la red profunda baja de 4.09 a 2.68 GB.
    POLY_MODULUS_MAX = 32768


    # Grado por defecto del polinomio de activación. Se opta por 7 siempre que se
    # puede: la profundidad que consume es ceil(log2(d+1)), así que 5 y 7 gastan
    # los mismos 3 niveles y el 7 aproxima mejor la ReLU al mismo coste.
    GRADO_PREFERIDO = 7

    def __init__(self, data_dir, model_path, config_path, degree=GRADO_PREFERIDO,
                 verbose=True):
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
        
        # 3. Aproximación polinómica de la ReLU AJUSTADA AL RANGO REAL de las
        # pre-activaciones (no a [-1,1] fijo): por el fan-in y las features
        # estandarizadas ese rango es mucho más ancho que [-1,1], y un polinomio
        # ajustado en [-1,1] se evaluaría fuera de su dominio (extrapolación).
        if degree == "sqt":
            # Activación cuadrada pura (no es ReLU): coeficientes fijos x^2.
            self.rango_activacion = None
            self.taylor_coeffs = [0.0, 0.0, 1.0]
        else:
            self.rango_activacion = self._leer_rango_preactivaciones()
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
        
        
    
    @staticmethod
    def profundidad_activacion(grado):
        """
        Niveles multiplicativos que consume evaluar el polinomio de activación.
        Para `polyval` de grado d el circuito mínimo tiene profundidad
        ceil(log2(d+1)), medido sobre TenSEAL 0.3.17: grados 2 y 3 gastan 2
        niveles, 5 y 7 gastan 3, y 9 gasta 4. De ahí que subir de grado 5 a 7 sea
        gratis en profundidad. 'sqt' (x^2) gasta 1.
        """
        if grado == "sqt":
            return 1
        return math.ceil(math.log2(int(grado) + 1))

    def profundidad_circuito(self):
        """
        Niveles que consume la inferencia cifrada completa de ESTA red: uno por
        capa lineal (cada `mm`) más una activación entre capas. Sumar el bias no
        gasta ninguno y la capa de salida no lleva activación, así que con una
        sola capa lineal el grado del polinomio es irrelevante.
        """
        num_lineales = len(self.pesos_lista)
        activaciones = max(0, num_lineales - 1)
        return num_lineales + activaciones * self.profundidad_activacion(self.degree)

    def presupuesto_bits(self, poly_modulus_degree):
        """
        Bits que admite la cadena de módulos de ese anillo sin bajar del nivel de
        seguridad estándar. Es la cota, no un gasto.
        """
        return self.MAX_BITS_SEGURIDAD[poly_modulus_degree][self.SEGURIDAD_MINIMA]

    @classmethod
    def seguridad_efectiva(cls, poly_modulus_degree, total_bits):
        """
        Nivel de seguridad que alcanza de verdad una cadena de `total_bits` en ese
        anillo: el mayor de la tabla cuya cota no se supera.

        Como la cota es un techo, gastar menos bits de los disponibles deja el
        nivel por encima del estándar. Una red cuya cadena ocupa 570 de los 881
        bits de N=32768 no está a 128 bits: está por debajo también de la cota de
        192 (611), así que alcanza ese nivel sin coste añadido.
        """
        cotas = cls.MAX_BITS_SEGURIDAD.get(poly_modulus_degree)
        if not cotas:                     # anillo fuera de la tabla del estándar
            return None
        alcanzados = [nivel for nivel, cota in cotas.items() if total_bits <= cota]
        return max(alcanzados) if alcanzados else None

    def config_maxima_precision(self):
        """
        Configuración CKKS óptima para esta red, o None si no hay ninguna viable.

        Los tres parámetros no se pueden elegir por separado, porque se reparten
        un mismo presupuesto de bits. El reparto se resuelve en este orden:

        1. PROFUNDIDAD. El circuito de la red fija cuántos primos intermedios
           necesita la cadena: uno por capa lineal más los de cada activación.
        2. SEGURIDAD. Ese número de primos hay que meterlo dentro del total de
           bits que admite cada anillo sin bajar del nivel estándar de 128 bits.
           La seguridad no gasta presupuesto, lo acota; lo repartible es lo que
           queda tras descontar los dos primos de los extremos, que no aportan
           profundidad. Quedarse por debajo de la cota eleva el nivel efectivo,
           que se calcula al final (seguridad_efectiva).
        3. PRECISIÓN. Dentro de ese margen se busca la escala más alta posible,
           que es de lo que depende la precisión del resultado:

               escala = (presupuesto - 2*60) / nº de primos intermedios

           acotada por ESCALA_MAXIMA, porque el primer primo tiene que alojar
           también la parte entera.
        4. COSTE. Entre los anillos que alcanzan ESCALA_OBJETIVO se escoge el
           más pequeño, que es el más barato de evaluar. El coste se optimiza al
           final, no como objetivo principal: si ninguno llega al objetivo, se
           devuelve el que mejor escala dé.
        """
        # 1. PROFUNDIDAD
        n_primos = self.profundidad_circuito()
        mejor = None

        for poly in self.POLY_CANDIDATOS:
            if poly > self.POLY_MODULUS_MAX:
                break
            # 2. SEGURIDAD
            disponible = self.presupuesto_bits(poly) - 2 * self.PRIMO_ESPECIAL
            if disponible <= 0:
                continue
            # 3. PRECISIÓN
            escala = min(self.ESCALA_MAXIMA, disponible // n_primos)
            if escala < self.ESCALA_MINIMA_POR_POLY.get(poly, 0):
                continue                      # con este anillo no hay primos así
            if mejor is None or escala > mejor[1]:
                mejor = (poly, escala)
            # 4. COSTE
            if escala >= self.ESCALA_OBJETIVO:
                mejor = (poly, escala)
                break                         # el más pequeño que sirve

        if mejor is None:
            return None

        poly, escala = mejor
        cadena = [self.PRIMO_ESPECIAL] + [escala] * n_primos + [self.PRIMO_ESPECIAL]
        return {
            'poly_modulus_degree': poly,
            'coeff_mod_bit_sizes': cadena,
            'global_scale_bits': escala,
            'grado': self.degree,
            'n_primos_medios': n_primos,
            'niveles_necesarios': n_primos,
            'total_bits': sum(cadena),
            'bits_disponibles': self.presupuesto_bits(poly),
            'seguridad_efectiva': self.seguridad_efectiva(poly, sum(cadena)),
            'margen_entero': 2 ** (self.PRIMO_ESPECIAL - escala),
            'alcanza_objetivo': escala >= self.ESCALA_OBJETIVO,
            'etiqueta': f"g{self.degree}_s{escala}_N{poly}",
        }

    def setup_encryption_context(self, poly_modulus_degree=None, coeff_mod_bit_sizes=None,
                                 global_scale=None):
        """
        Configura el contexto de cifrado CKKS para operaciones homomórficas.

        Los tres parámetros del esquema se pueden dar o dejar que los calcule la
        propia clase:

        - Si se dan los tres, se usan tal cual. Es lo que hace el barrido, que
          necesita forzar configuraciones concretas para poder compararlas.
        - Si no se dan, se derivan de ESTA red: las capas lineales ya están en
          self.pesos_lista y el grado del polinomio en self.degree, que es el
          mismo con el que se ajustaron los coeficientes. La cadena se dimensiona
          a la profundidad real del circuito y la escala se lleva al máximo que
          admite el presupuesto de bits (ver configuracion_ckks).

        Args:
            poly_modulus_degree: grado del polinomio del anillo (mayor = más lento).
            coeff_mod_bit_sizes: tamaños en bits de la cadena de módulos (define la
                                 profundidad multiplicativa disponible).
            global_scale: escala de codificación de los números en punto flotante.

        Deja en self.ckks la configuración realmente usada, para que quien guarde
        los resultados no tenga que reconstruirla.
        """
        dados = (poly_modulus_degree, coeff_mod_bit_sizes, global_scale)
        if any(p is None for p in dados):
            if not all(p is None for p in dados):
                raise ValueError(
                    "setup_encryption_context: o se dan poly_modulus_degree, "
                    "coeff_mod_bit_sizes y global_scale a la vez, o ninguno. Mezclar "
                    "una escala con una cadena que no le corresponde rompe el reescalado.")
            cfg = self.config_maxima_precision()
            if cfg is None:
                raise ValueError(
                    f"No hay configuración CKKS viable para {len(self.pesos_lista)} capas "
                    f"lineales con polinomio de grado {self.degree}: el circuito es "
                    f"demasiado profundo para el presupuesto de bits disponible. Baja el "
                    f"grado o sube POLY_MODULUS_MAX.")
            poly_modulus_degree = cfg['poly_modulus_degree']
            coeff_mod_bit_sizes = cfg['coeff_mod_bit_sizes']
            global_scale = 2 ** cfg['global_scale_bits']

        # Configuración efectiva, calculada o recibida. La seguridad se mide sobre
        # la cadena que se va a usar, venga de donde venga: es una propiedad del
        # total de bits, no de cómo se hayan elegido.
        self.ckks = {
            'poly_modulus_degree': poly_modulus_degree,
            'coeff_mod_bit_sizes': coeff_mod_bit_sizes,
            'global_scale_bits': int(round(math.log2(global_scale))),
            'grado': self.degree,
            'seguridad_efectiva': self.seguridad_efectiva(poly_modulus_degree,
                                                          sum(coeff_mod_bit_sizes)),
        }

        if self.verbose:
            print("\n🔐 Configurando contexto de cifrado homomórfico...")
            print(f"   Esquema: CKKS (permite operaciones con punto flotante)")
            print(f"   Poly modulus degree: {poly_modulus_degree}")
            print(f"   Cadena: {coeff_mod_bit_sizes}")
            print(f"   Escala: 2^{self.ckks['global_scale_bits']}")

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
        #self.context.generate_galois_keys()

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

    def _leer_rango_preactivaciones(self):
        """
        Devuelve el rango [a, b] de las pre-activaciones sobre el que ajustar el
        polinomio, leído de model_config.json.

        Lo mide Trainer.last_train sobre train+val con los pesos ya definitivos, y
        acompaña al modelo como un parámetro más: quien infiere sobre el cifrado no
        necesita ver ningún dato en claro para reconstruir la activación. El rango
        se midió con la ReLU real y el polinomio la aproxima sobre [a, b], así que
        sigue siendo válido en la inferencia cifrada.

        Devuelve None si la red no tiene capas ocultas: no hay activación que
        aproximar.
        """
        rango = self.config.get('rango_preactivaciones')
        if rango is None:
            if self.model.hidden_layers:
                raise ValueError(
                    "model_config.json no incluye 'rango_preactivaciones' y la red "
                    "tiene capas ocultas: vuelve a entrenar el modelo para medirlo."
                )
            return None
        a, b = rango
        return float(a), float(b)

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

