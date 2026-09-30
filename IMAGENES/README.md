# 🔬 Guía de Entrenamiento de Redes Neuronales para Detección de Anemia Infantil

Este directorio contiene los cuadernos de Jupyter, datos y configuraciones desarrollados para la tesis:
> **"MÉTODO NO INVASIVO BASADO EN REDES NEURONALES CONVOLUCIONALES PARA LA DETECCIÓN DE LA ANEMIA INFANTIL A PARTIR DE IMÁGENES EN EL CENTRO DE SALUD, OCOBAMBA 2025"**  
> *Universidad Nacional José María Arguedas (UNAJMA) - 2026*

---

## 📁 Estructura del Proyecto

```
C:\Proyectos\luu\
├── .claude\
│   └── skills\
│       ├── anemia-vision-studio\        # Skill Maestro especializado para la tesis
│       ├── deep-learning-pytorch\       # Skill estándar PyTorch (skills.sh)
│       ├── computer-vision-opencv\      # Skill estándar Visión Computacional (skills.sh)
│       └── automl-hyperparameter-optimization\ # Skill estándar AutoML / Optuna (skills.sh)
├── IMAGENES\
│   ├── Dataset\                         # Dataset real de conjuntiva palpebral
│   │   ├── images\                      # 888 imágenes médicas
│   │   ├── train.csv                    # 606 casos de entrenamiento
│   │   ├── validation.csv               # 145 casos de validación
│   │   ├── test.csv                     # 137 casos de prueba final (no vistos)
│   │   └── metadata.csv                 # Metadatos clínicos (HB, Edad, Sexo)
│   ├── ResNet-50.ipynb                  # Cuaderno de entrenamiento de ResNet-50
│   ├── EfficientNet-B3.ipynb            # Cuaderno de entrenamiento de EfficientNet-B3
│   ├── results\                         # Gráficas de alta resolución y CSVs generados
│   │   ├── resnet50_metricas_test.csv
│   │   ├── resnet50_graficas_tesis.png
│   │   ├── efficientnet_b3_metricas_test.csv
│   │   ├── efficientnet_b3_graficas_tesis.png
│   │   └── tabla_comparativa_modelos_tesis.csv
│   ├── best_resnet50.pth                # Checkpoint con los mejores pesos
│   └── best_efficientnet_b3.pth         # Checkpoint con los mejores pesos
└── tesis.docx                           # Documento de investigación de la tesis
```

---

## 🚀 Cómo Ejecutar los Modelos

### Opción 1: En VS Code o Jupyter Notebook / JupyterLab
1. Abra `C:\Proyectos\luu\IMAGENES\ResNet-50.ipynb`.
2. Puede ajustar hiperparámetros en la **Celda 2** (`CONFIG`):
   ```python
   CONFIG = {
       'IMAGE_SIZE': 224,
       'BATCH_SIZE': 16,
       'EPOCHS': 25,
       'LEARNING_RATE': 1e-4,
       'WEIGHT_DECAY': 1e-4,
       'DROPOUT_RATE': 0.4
   }
   ```
3. Ejecute las celdas en orden (o presione **Run All / Ejecutar Todo**).
4. El cuaderno automáticamente:
   - Aplica Data Augmentation biomédico adaptado a conjuntiva palpebral.
   - Pondera la función de pérdida para balancear clases (346 Anemia vs 260 No anemia).
   - Guarda el mejor modelo en `best_resnet50.pth`.
   - Evalúa en los 137 casos de `test.csv`.
   - Genera la Matriz de Confusión, Curva ROC (AUC), y métricas clínicas de **Sensibilidad** y **Especificidad**.
   - Guarda los gráficos en `results/resnet50_graficas_tesis.png`.
5. Abra y ejecute `C:\Proyectos\luu\IMAGENES\EfficientNet-B3.ipynb`.
6. En la última celda de `EfficientNet-B3.ipynb`, se generará automáticamente el **Cuadro Comparativo** entre ResNet-50 y EfficientNet-B3 listo para insertar en el **Capítulo VI** de su tesis.

---

## 📊 Métricas Clínicas Evaluadas para la Tesis

| Métrica Clínica | Significado Científico | Relevancia en Salud Infantil |
| :--- | :--- | :--- |
| **Sensibilidad (Recall)** | $TP / (TP + FN)$ | **Crítica**: Evita que niños con anemia sean diagnosticados erróneamente como sanos (falsos negativos). |
| **Especificidad** | $TN / (TN + FP)$ | Capacidad de clasificar correctamente a pacientes sin anemia (evita tratamientos innecesarios). |
| **Exactitud (Accuracy)** | $(TP + TN) / Total$ | Porcentaje global de aciertos del modelo. |
| **Precisión (VPP)** | $TP / (TP + FP)$ | Probabilidad de que un niño con predicción positiva realmente tenga anemia. |
| **F1-Score** | $2 \cdot \frac{Prec \cdot Sens}{Prec + Sens}$ | Balance armónico entre precisión y sensibilidad ante clases no perfectamente balanceadas. |
| **Curva ROC y AUC** | Área Bajo la Curva | Capacidad discriminativa del modelo a distintos umbrales de decisión. |

---

## 🧠 Skills Instalados en Claude Code
Este proyecto cuenta con los skills oficiales integrados de **[skills.sh](https://www.skills.sh)**:
- `deep-learning-pytorch`: Buenas prácticas de programación en PyTorch, GPU y precisión mixta.
- `computer-vision-opencv`: Procesamiento y aumentos de imágenes biomédicas.
- `automl-hyperparameter-optimization`: Optimización de hiperparámetros con Optuna y espacios logarítmicos.
- `anemia-vision-studio`: Skill maestro del proyecto para diagnosticar y ajustar los modelos de la tesis.
