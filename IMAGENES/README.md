# Guia de Entrenamiento de Redes Neuronales para Deteccion de Anemia Infantil

Este directorio contiene los cuadernos de Jupyter, datos y configuraciones desarrollados para la tesis:
> **"METODO NO INVASIVO BASADO EN REDES NEURONALES CONVOLUCIONALES PARA LA DETECCION DE LA ANEMIA INFANTIL A PARTIR DE IMAGENES EN EL CENTRO DE SALUD, OCOBAMBA 2025"**  
> *Universidad Nacional Jose Maria Arguedas (UNAJMA) - 2026*

---

## Estructura del Proyecto

```
C:\Proyectos\luu\
├── .claude\
│   └── skills\
│       ├── anemia-vision-studio\        # Skill Maestro especializado para la tesis
│       ├── deep-learning-pytorch\       # Skill estandar PyTorch (skills.sh)
│       ├── computer-vision-opencv\      # Skill estandar Vision Computacional (skills.sh)
│       └── automl-hyperparameter-optimization\ # Skill estandar AutoML / Optuna (skills.sh)
├── IMAGENES\
│   ├── Dataset\                         # Dataset real de conjuntiva palpebral
│   │   ├── images\                      # 888 imagenes medicas
│   │   ├── train.csv                    # 606 casos de entrenamiento
│   │   ├── validation.csv               # 145 casos de validacion
│   │   ├── test.csv                     # 137 casos de prueba final (no vistos)
│   │   └── metadata.csv                 # Metadatos clinicos (HB, Edad, Sexo)
│   ├── ResNet-50.ipynb                  # Cuaderno de entrenamiento de ResNet-50
│   ├── EfficientNet-B3.ipynb            # Cuaderno de entrenamiento de EfficientNet-B3
│   ├── results\                         # Graficas de alta resolucion y CSVs generados
│   │   ├── resnet50_metricas_test.csv
│   │   ├── resnet50_graficas_tesis.png
│   │   ├── efficientnet_b3_metricas_test.csv
│   │   ├── efficientnet_b3_graficas_tesis.png
│   │   └── tabla_comparativa_modelos_tesis.csv
│   ├── best_resnet50.pth                # Checkpoint con los mejores pesos
│   └── best_efficientnet_b3.pth         # Checkpoint con los mejores pesos
└── tesis.docx                           # Documento de investigacion de la tesis
```

---

## Instrucciones para Ejecutar los Modelos

### En VS Code o Jupyter Notebook / JupyterLab
1. Abra `C:\Proyectos\luu\IMAGENES\ResNet-50.ipynb`.
2. Puede ajustar hiperparametros en la **Celda 2** (`CONFIG`):
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
3. Ejecute las celdas en orden secuencial (o seleccione **Run All / Ejecutar Todo**).
4. El cuaderno automaticamente:
   - Aplica Data Augmentation biomedico adaptado a conjuntiva palpebral.
   - Pondera la funcion de perdida para balancear clases (346 Anemia vs 260 No anemia).
   - Guarda el mejor modelo en `best_resnet50.pth`.
   - Evalua en los 137 casos de `test.csv`.
   - Genera la Matriz de Confusion, Curva ROC (AUC), y metricas clinicas de **Sensibilidad** y **Especificidad**.
   - Guarda los graficos en `results/resnet50_graficas_tesis.png`.
5. Abra y ejecute `C:\Proyectos\luu\IMAGENES\EfficientNet-B3.ipynb`.
6. En la ultima celda de `EfficientNet-B3.ipynb`, se generara automaticamente el **Cuadro Comparativo** entre ResNet-50 y EfficientNet-B3 listo para insertar en el **Capitulo VI** de su tesis.

---

## Metricas Clinicas Evaluadas para la Tesis

| Metrica Clinica | Significado Cientifico | Relevancia en Salud Infantil |
| :--- | :--- | :--- |
| **Sensibilidad (Recall)** | $TP / (TP + FN)$ | **Critica**: Evita que ninos con anemia sean diagnosticados erroneamente como sanos (falsos negativos). |
| **Especificidad** | $TN / (TN + FP)$ | Capacidad de clasificar correctamente a pacientes sin anemia (evita tratamientos innecesarios). |
| **Exactitud (Accuracy)** | $(TP + TN) / Total$ | Porcentaje global de aciertos del modelo. |
| **Precision (VPP)** | $TP / (TP + FP)$ | Probabilidad de que un nino con prediccion positiva realmente tenga anemia. |
| **F1-Score** | $2 \cdot \frac{Prec \cdot Sens}{Prec + Sens}$ | Balance armonico entre precision y sensibilidad ante clases no perfectamente balanceadas. |
| **Curva ROC y AUC** | Area Bajo la Curva | Capacidad discriminativa del modelo a distintos umbrales de decision. |

---

## Skills Instalados en Claude Code
Este proyecto cuenta con los skills oficiales integrados de **[skills.sh](https://www.skills.sh)**:
- `deep-learning-pytorch`: Buenas practicas de programacion en PyTorch, GPU y precision mixta.
- `computer-vision-opencv`: Procesamiento y aumentos de imagenes biomedicas.
- `automl-hyperparameter-optimization`: Optimizacion de hiperparametros con Optuna y espacios logaritmicos.
- `anemia-vision-studio`: Skill maestro del proyecto para diagnosticar y ajustar los modelos de la tesis.
