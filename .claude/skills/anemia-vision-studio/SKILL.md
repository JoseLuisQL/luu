---
name: anemia-vision-studio
description: Skill especializado para construcción, configuración y entrenamiento de modelos de redes neuronales convolucionales (ResNet-50 y EfficientNet-B3) aplicados a la detección de anemia infantil a partir de imágenes de conjuntiva palpebral para la tesis (UNAJMA 2026).
---

# Anemia Vision Studio - Skill Especializado para Tesis de Deep Learning

Este skill instruye a Claude Code para construir, configurar, diagnosticar y evaluar modelos de redes neuronales convolucionales con PyTorch, integrando las mejores prácticas de **skills.sh** (`deep-learning-pytorch`, `computer-vision-opencv`, `automl-hyperparameter-optimization`) con el contexto científico de la tesis:
*"MÉTODO NO INVASIVO BASADO EN REDES NEURONALES CONVOLUCIONALES PARA LA DETECCIÓN DE LA ANEMIA INFANTIL A PARTIR DE IMÁGENES EN EL CENTRO DE SALUD, OCOBAMBA 2025"*.

## Ubicación del Proyecto y Recursos
- **Notebooks de Ejecución**:
  - `C:\Proyectos\luu\IMAGENES\ResNet-50.ipynb` (Entrenamiento y evaluación de ResNet-50)
  - `C:\Proyectos\luu\IMAGENES\EfficientNet-B3.ipynb` (Entrenamiento y evaluación de EfficientNet-B3)
- **Dataset Real**:
  - `C:\Proyectos\luu\IMAGENES\Dataset\images\` (Directorio de imágenes)
  - `C:\Proyectos\luu\IMAGENES\Dataset\train.csv` (606 casos: 346 Anemia, 260 No Anemia)
  - `C:\Proyectos\luu\IMAGENES\Dataset\validation.csv` (145 casos: 94 Anemia, 51 No Anemia)
  - `C:\Proyectos\luu\IMAGENES\Dataset\test.csv` (137 casos: 69 Anemia, 68 No Anemia)
- **Marco de Tesis**: `C:\Proyectos\luu\tesis.docx`

---

## 1. Protocolo de Arquitectura y Transfer Learning

### A. ResNet-50 (`ResNet-50.ipynb`)
- **Resolución de Entrada**: `224x224` píxeles.
- **Pesos Preentrenados**: `torchvision.models.ResNet50_Weights.DEFAULT`.
- **Estructura del Clasificador**:
  ```python
  model.fc = nn.Sequential(
      nn.Linear(2048, 512),
      nn.BatchNorm1d(512),
      nn.ReLU(inplace=True),
      nn.Dropout(0.4),
      nn.Linear(512, 2)
  )
  ```

### B. EfficientNet-B3 (`EfficientNet-B3.ipynb`)
- **Resolución de Entrada**: `300x300` píxeles (resolución nativa óptima para escalado compuesto de B3).
- **Pesos Preentrenados**: `torchvision.models.EfficientNet_B3_Weights.DEFAULT`.
- **Estructura del Clasificador**:
  ```python
  model.classifier = nn.Sequential(
      nn.Dropout(0.3),
      nn.Linear(1536, 256),
      nn.SiLU(inplace=True),
      nn.Dropout(0.2),
      nn.Linear(256, 2)
  )
  ```

---

## 2. Protocolo de Data Augmentation Biomédico
El color de la conjuntiva palpebral (eritema / palidez) es el biomarcador visual clave de la hemoglobina. Por ende:
- **Permitido**:
  - `RandomHorizontalFlip(p=0.5)`
  - `RandomRotation(degrees=(-10, 10))`
  - `RandomResizedCrop(size, scale=(0.85, 1.0))`
  - `ColorJitter(brightness=0.1, contrast=0.1)` (pequeñas variaciones de iluminación de cámara).
- **Prohibido o Restringido**:
  - NO usar transformaciones extremas de tono (`hue`) que alteren el color de la sangre o conjuntiva.
  - Normalización estándar de ImageNet: `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.

---

## 3. Manejo de Desbalance y Función de Pérdida
En el conjunto de entrenamiento hay 346 casos positivos (Anemia) y 260 casos negativos (No anemia):
```python
total = 606
w0 = total / (2.0 * 260)  # ~1.165 para No Anemia
w1 = total / (2.0 * 346)  # ~0.876 para Anemia
weights = torch.tensor([w0, w1], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
```

---

## 4. Métricas Clínicas Exigidas para la Tesis (Capítulos V y VI)
Para cada modelo en el conjunto de prueba (`test.csv` de 137 muestras):
1. **Exactitud (Accuracy)**: $(TP + TN) / (TP + TN + FP + FN)$
2. **Sensibilidad (Recall / Sensibilidad Diagnóstica)**:
   $$Sensibilidad = \frac{TP}{TP + FN}$$
   *(Métrica más crítica en salud para evitar falsos negativos en niños con anemia).*
3. **Especificidad**:
   $$Especificidad = \frac{TN}{TN + FP}$$
4. **Precisión (Valor Predictivo Positivo)**: $TP / (TP + FP)$
5. **F1-Score**: $2 \cdot \frac{Precisión \cdot Sensibilidad}{Precisión + Sensibilidad}$
6. **Curva ROC y AUC (Área Bajo la Curva)**
7. **Matriz de Confusión Normalizada y en Cantidades**

---

## 5. Salidas Requeridas para el Documento de Tesis
- Guardar los mejores pesos en `best_resnet50.pth` y `best_efficientnet_b3.pth`.
- Gráficas de curvas de aprendizaje: `train_loss` vs `val_loss` y `train_acc` vs `val_acc` por época.
- Tabla comparativa final estructurada para el Capítulo VI ("Comparación de modelos de redes neuronales").
