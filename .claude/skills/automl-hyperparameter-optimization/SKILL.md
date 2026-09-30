---
name: automl-hyperparameter-optimization
description: Optimización de hiperparámetros, búsqueda bayesiana con Optuna, diseño de espacios de búsqueda y validación de modelos (origen skills.sh / mindrally).
---

# AutoML & Hyperparameter Optimization (skills.sh Standard)

## Principios Fundamentales
1. **Espacios de Búsqueda Logarítmicos**:
   - Tasa de aprendizaje (`learning_rate`): Muestrear en escala logarítmica, e.g., `trial.suggest_float('lr', 1e-5, 1e-2, log=True)`.
   - Decaimiento de peso (`weight_decay`): `trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)`.
   - Regularización y Dropout: Escala lineal, e.g., `trial.suggest_float('dropout', 0.2, 0.5, step=0.1)`.
   - Tamaño de lote (`batch_size`): Selección categórica de potencias de 2, e.g., `trial.suggest_categorical('batch_size', [8, 16, 32])`.

2. **Estrategias de Pruning y Aceleración**:
   - Monitorear la función objetivo en validación (e.g., `val_loss` o `val_f1`).
   - Usar pruners medianos o hiperband para descartar ensayos no prometedores tempranamente.

3. **Manejo de Desbalance de Clases**:
   - Cuando las clases no estén en proporción 1:1, calcular pesos inversos a la frecuencia:
     `weight_c = total_samples / (n_classes * samples_c)`.
   - Inyectar estos pesos en la función de pérdida (`nn.CrossEntropyLoss(weight=weights)`).
