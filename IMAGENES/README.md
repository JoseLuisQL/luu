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
│   ├── Dataset.zip                      # Archivo comprimido listo para Google Colab (70.58 MB)
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

### Escenario A: En su Computadora Local (VS Code, JupyterLab o Anaconda)
1. Abra `ResNet-50.ipynb` o `EfficientNet-B3.ipynb`.
2. El cuaderno localiza automaticamente la carpeta `Dataset` en su sistema.
3. Ejecute las celdas en orden secuencial (o seleccione **Run All / Ejecutar Todo**).
4. El cuaderno automaticamente:
   - Aplica Data Augmentation biomedico adaptado a conjuntiva palpebral.
   - Pondera la funcion de perdida para balancear clases (346 Anemia vs 260 No anemia).
   - Guarda el mejor modelo en `best_resnet50.pth` o `best_efficientnet_b3.pth`.
   - Evalua en los 137 casos de `test.csv`.
   - Genera la Matriz de Confusion, Curva ROC (AUC), y metricas de **Sensibilidad** y **Especificidad**.
   - Guarda los graficos en `results/`.
5. Al ejecutar ambos cuadernos, la celda final de `EfficientNet-B3.ipynb` genera el **Cuadro Comparativo** para el **Capitulo VI** de su tesis.

### Escenario B: En Google Colab o Kaggle (GPU en la Nube)
1. Suba el archivo `ResNet-50.ipynb` a Google Colab.
2. En el panel izquierdo de archivos de Colab (icono de carpeta):
   - Arrastre y suba el archivo `Dataset.zip` que ya fue generado en `C:\Proyectos\luu\IMAGENES\Dataset.zip`.
3. Ejecute la **Celda 0** del cuaderno para descomprimir el dataset:
   ```bash
   !unzip -q -o Dataset.zip
   ```
4. El localizador automatico del cuaderno detectara `/content/Dataset` y procedera con el entrenamiento en GPU acelerada (T4/V100/A100).

## Fases de la Metodologia KDD Implementadas en los Cuadernos

Los cuadernos `ResNet-50.ipynb` y `EfficientNet-B3.ipynb` implementan de forma rigurosa las fases de la metodologia KDD (Knowledge Discovery in Databases) estipulada en la tesis:

1. **Fase 1: Seleccion y EDA Clinico:** Analisis exploratorio de hemoglobina, edad en meses y distribucion demografica bajo criterios de la OMS, exportando la Tabla 1 y Figura 1 a 300 DPI segun norma APA 7ma edicion.
2. **Fase 2: Preprocesamiento y Limpieza:** Cargador robusto para imagenes biomedicas que resuelve incidencias de CRC en chunks PNG (como `T_A95_img_002.png`), fusion de canales RGBA sobre fondo blanco neutro y saneamiento de etiquetas contradictorias para erradicar cualquier fuga de datos (*data leakage*).
3. **Fase 3: Transformacion y Data Augmentation:** Normalizacion segun ImageNet, aumentos leves que preservan el matiz cromatico vascular de la conjuntiva palpebral y generacion de mosaicos clinicos comparativos a 300 DPI (224x224 px para ResNet-50 y 300x300 px para EfficientNet-B3).
4. **Fase 4: Mineria de Datos y Modelado en Dos Fases:** Transfer Learning con arquitecturas convolucionales avanzadas (ResNet-50 de 24.5M parametros y EfficientNet-B3 de 12.2M parametros con MBConv y Squeeze-and-Excitation), aplicando Warmup de 5 epocas y Fine-Tuning progresivo de 20 epocas con decaimiento cosenoidal (`CosineAnnealingLR`) y perdida balanceada por clases.
5. **Fase 5: Evaluacion Clinica e Interpretacion:** Evaluacion rigurosa en el conjunto de prueba independiente (`test.csv`, n = 137), calibracion del umbral clinico optimo de Youden ($J = Sensibilidad + Especificidad - 1$) para minimizar falsos negativos y falsos positivos, calculo de metricas con intervalos de confianza al 95% (Wilson) y graficos academicos a 300 DPI (curvas de aprendizaje, matrices de confusion y ROC-AUC).
6. **Fase 6: Despliegue e Interoperabilidad:** Exportacion validada a formato abierto ONNX (`.onnx`) y TorchScript (`.pt`) con verificacion de paridad numerica ($Error < 10^{-4}$) para su integracion en el aplicativo movil del centro de salud.
7. **Protocolo Multiejecucion y Contraste Estadistico:** Evaluacion con 5 semillas independientes ([42, 101, 202, 303, 404]) para certificar la repetibilidad clinica (Corcuff & Leveque, 2006) y contraste formal de hipotesis mediante pruebas t de Student y Wilcoxon para sustentar el Capitulo VI de la tesis.

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
