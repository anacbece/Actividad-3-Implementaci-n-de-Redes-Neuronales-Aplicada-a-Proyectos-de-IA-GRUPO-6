#Actividad 3: Implementación de Redes Neuronales Aplicada a Proyecto Final

Descripción General
Esta actividad implementa redes neuronales desde cero usando NumPy, demostrando comprensión profunda de:

*Propagación hacia adelante (forward)
*Propagación hacia atrás (backpropagation)
*Inicialización de pesos (Xavier/He)
*Funciones de activación (ReLU, Sigmoid, Tanh)
*Entrenamiento usando gradiente descendente
*Predicción para un caso aplicado al Proyecto Final: modelo simple de recomendación (Neural Matrix Factorization – MF)
*Adicionalmente, se realizan experimentos comparativos entre tres arquitecturas neurales y un modelo baseline lineal, evaluando su rendimiento mediante RMSE.

Contenido del Proyecto
El script contiene 4 partes principales:

Parte 1 — Red Neuronal desde Cero (NumPy)
Estructura Base (NeuralNetwork)
Múltiples capas ocultas
Inicialización Xavier/He
Funciones de activación intercambiables:

  *relu
  *sigmoid
  *tanh

Forward propagation
Backpropagation manual
Actualización de pesos con gradiente descendente

Componentes Implementados
Pesos y bias con NumPy	✔
Propagación hacia adelante	✔
Regla delta (backprop)	✔
Derivadas de activaciones	✔
Cálculo de loss MSE	✔
Desacoplamiento modular	✔

Parte 2 — Aplicación al Proyecto Final (Sistema de Recomendación)
Se implementa una versión simplificada de:

✔ Neural Matrix Factorization (NumPy)

Embeddings de usuario e ítem

Capa MLP opcional

Predicción de ratings entre 1–5

✔ Datos utilizados

Si no se encuentra ratings.csv en MovieLens:

→ Se genera un dataset simulado, con:

5000 usuarios

2000 ítems

200,000 interacciones

Esto permite entrenar sin necesidad de cargar MovieLens completo en NumPy.

Parte 3 — Experimentación Comparativa

Se prueban tres arquitecturas neurales + baseline lineal:

*Modelo A_small

  Embedding: 8
  
  MLP: [16]
  
  Activación: ReLU

*Modelo B_medium

  Embedding: 16
  
  MLP: [32, 16]
  
  Activación: ReLU

*Modelo C_wide

  Embedding: 32
  
  MLP: [64, 32]
  
  Activación: Tanh

*Baseline lineal

  Modelo de regresión lineal simple.

*Métrica utilizada

  RMSE (Root Mean Squared Error)

Resultados Obtenidos
Modelo	RMSE
A_small	0.95269
B_medium	0.95267
C_wide	0.95269
Baseline lineal	0.95270

Todos los modelos obtuvieron valores muy similares.
Esto ocurre porque los datos son simulados → no tienen estructura latente fuerte, por lo que los modelos complejos no pueden aprender relaciones reales.

Parte 4 — Conexión con el Proyecto Final

* Limitaciones de la implementación NumPy

No escala a datasets reales (25M interacciones).

No usa GPU.

Backpropagation es demasiado lento para producción.

Sin batching eficiente.

Sin optimizadores modernos (Adam, RMSProp).

* Por qué usar PyTorch en el proyecto final

Maneja embeddings enormes

Entrenamiento en GPU

DataLoaders eficientes

AMP (mixed precision)

Autograd

Modelos híbridos (MF + MLP + secuencias)

# Instrucciones de Ejecución
# Requisitos
numpy
matplotlib
scikit-learn

# Ejecutar el script

Desde notebook o Python estándar:

from numpy_neural_mf import run_experiments
run_experiments()

# Salida generada

Entrenamiento de cada arquitectura

Loss por época

RMSE final

Gráfico comparativo automático (comparacion_arquitecturas.png)

# Estructura del Repositorio
actividad3/
 ├── numpy_neural_mf.py
 ├── comparacion_arquitecturas.png
 ├── README.md

# Conclusiones Generales

Las redes neuronales hechas a mano funcionan correctamente (validación educativa).

Las arquitecturas más grandes no mejoraron el RMSE debido a datos simulados.

La implementación NumPy NO es adecuada para MovieLens → se requiere PyTorch para el proyecto final.

La actividad mantiene coherencia total con el Proyecto 3 (Recomendador Híbrido).
