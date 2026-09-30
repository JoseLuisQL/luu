---
name: deep-learning-pytorch
description: Flujos de trabajo avanzados de Deep Learning, arquitecturas modulares OOP, optimización en PyTorch, precisión mixta y entrenamiento reproducible (origen skills.sh / mindrally).
---

# Deep Learning con PyTorch (skills.sh Standard)

## Principios Fundamentales
1. **Arquitectura Orientada a Objetos (OOP)**:
   - Todo modelo hereda de `torch.nn.Module`.
   - Inicialización explícita de capas en `__init__` con pesos configurados (`kaiming_normal_`, `xavier_uniform_`).
   - El método `forward(*args, **kwargs)` debe ser puramente funcional sobre los tensores.
   - Usar `register_buffer()` para tensores no entrenables (e.g., medias o máscaras estáticas).

2. **Gestión de Hardware y GPU**:
   - Detección dinámica: `device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')`.
   - Transferencia limpia: `tensor.to(device, non_blocking=True)`.
   - Evitar sincronizaciones bloqueantes innecesarias dentro de bucles de entrenamiento.

3. **Precisión Mixta Automática (AMP)**:
   - Cuando se utilice GPU, emplear `torch.amp.autocast('cuda')` y `torch.amp.GradScaler('cuda')`.
   - Fallback limpio a precisión simple (`float32`) en CPU.

4. **Pipelines de Datos Funcionales**:
   - Desacoplar la ingesta de datos con `torch.utils.data.Dataset` y `DataLoader`.
   - Usar `pin_memory=True` y `num_workers > 0` cuando haya aceleración por GPU.
   - Fijar semillas deterministas en PyTorch, NumPy y Random (`manual_seed`).

5. **Entrenamiento y Checkpointing**:
   - Mantener métricas de pérdida de entrenamiento y validación separadas por época.
   - Guardar el estado completo (`torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': val_loss}, path)`).
