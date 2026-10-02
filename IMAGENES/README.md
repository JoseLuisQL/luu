# Guia de Entrenamiento de Redes Neuronales para Deteccion de Anemia Infantil

Este directorio contiene los cuadernos de Jupyter, datos y configuraciones desarrollados para la tesis:
> **"METODO NO INVASIVO BASADO EN REDES NEURONALES CONVOLUCIONALES PARA LA DETECCION DE LA ANEMIA INFANTIL A PARTIR DE IMAGENES EN EL CENTRO DE SALUD, OCOBAMBA 2025"**  
> *Universidad Nacional Jose Maria Arguedas (UNAJMA) - 2026*

---

## Estructura del Proyecto

```text
C:\Proyectos\luu\
├── README.md                            # Guia principal y documentacion general del sistema
├── tesis.docx                           # Documento de investigacion de la tesis
├── .claude\
│   └── skills\
│       ├── anemia-vision-studio\        # Skill especializado para la tesis
│       ├── deep-learning-pytorch\       # Skill PyTorch
│       ├── computer-vision-opencv\      # Skill Vision Computacional
│       └── automl-hyperparameter-optimization\ # Skill Optimizacion de hiperparametros
└── IMAGENES\
    ├── README.md                        # Esta guia operativa
    ├── Dataset\                         # Dataset clinico de conjuntiva palpebral
    │   ├── images\                      # 888 imagenes medicas (IMG0001.png a IMG0888.png)
    │   ├── metadata.csv                 # 888 registros maestros (HB, Edad, Sexo)
    │   ├── train.csv                    # 606 casos de entrenamiento
    │   ├── validation.csv               # 145 casos de validacion interna
    │   └── test.csv                     # 137 casos de prueba independiente ciega
    ├── Dataset.zip                      # Archivo comprimido listo para Google Colab o Kaggle
    ├── ResNet-50.ipynb                  # Cuaderno de entrenamiento de ResNet-50
    ├── EfficientNet-B3.ipynb            # Cuaderno de entrenamiento de EfficientNet-B3
    ├── prueba-anemia-resnet.ipynb       # Cuaderno ResNet-50 optimizado para Kaggle
    ├── prueba-anemia-eficcient.ipynb    # Cuaderno EfficientNet-B3 optimizado para Kaggle
    ├── build_notebooks_definitivos.py   # Generador maestro de celdas y validacion
    ├── generate_notebooks.py            # Compilador de cuadernos
    ├── test_pipeline.py                 # Suite de pruebas unitarias
    ├── results\                         # Graficas de alta resolucion a 300 DPI y tablas CSV
    │   ├── resnet50_metricas_test.csv
    │   ├── resnet50_graficas_tesis.png
    │   ├── efficientnet_b3_metricas_test.csv
    │   ├── efficientnet_b3_graficas_tesis.png
    │   └── tabla_comparativa_modelos_tesis.csv
    ├── best_resnet50.pth                # Checkpoint con los mejores pesos
    └── best_efficientnet_b3.pth         # Checkpoint con los mejores pesos
```

---

## 1. Como Subir el Dataset a Kaggle Correctamente

Para evitar fallos de lectura de imagenes y carpetas anidadas no reconocidas, siga estos pasos:

### Paso 1: Preparacion del Archivo Comprimido
1. Ingrese a la carpeta `C:\Proyectos\luu\IMAGENES\Dataset\`.
2. Seleccione la carpeta `images` y los 4 archivos CSV (`metadata.csv`, `train.csv`, `validation.csv`, `test.csv`).
3. Comprima directamente estos elementos seleccionados en un archivo `.zip` (ejemplo: `dataset-anemia-ocobamba.zip`).
   *Importante*: Al abrir el archivo zip, la carpeta `images` y los 4 CSV deben figurar directamente en la raiz del archivo, sin una carpeta padre adicional.

### Paso 2: Publicacion del Dataset en Kaggle
1. Inicie sesion en su cuenta de [kaggle.com](https://www.kaggle.com).
2. En el menu lateral izquierdo, haga clic en **Datasets** y luego en el boton superior **+ New Dataset**.
3. En **Dataset Title**, coloque un nombre simple sin espacios ni caracteres especiales, por ejemplo: `dataset-anemia-oco`.
4. Arrastre el archivo `dataset-anemia-ocobamba.zip` (o el ya preparado `Dataset.zip`) a la zona de carga.
5. Haga clic en **Create**.
6. Kaggle procesara el archivo y lo dejara montado en el contenedor bajo la ruta:
   `/kaggle/input/dataset-anemia-oco/`

---

## 2. Como Configurar y Ejecutar los Cuadernos en Kaggle

### Paso 1: Crear o Importar el Cuaderno
1. En Kaggle, dirijase a **Code** -> **+ New Notebook**.
2. En el menu del cuaderno, seleccione **File** -> **Import Notebook**.
3. Cargue el archivo correspondiente desde su computadora:
   - Para ResNet-50: `prueba-anemia-resnet.ipynb` (o `ResNet-50.ipynb`)
   - Para EfficientNet-B3: `prueba-anemia-eficcient.ipynb` (o `EfficientNet-B3.ipynb`)

### Paso 2: Vincular el Dataset al Cuaderno
1. En el panel lateral derecho del cuaderno, abra la seccion **Input**.
2. Haga clic en **+ Add Data**.
3. En la pestana **Your Datasets**, busque `dataset-anemia-oco` y haga clic en el boton **+** para agregarlo al cuaderno.

### Paso 3: Configuraciones Criticas de la Sesion (Notebook Options)
En la barra lateral derecha, realice los siguientes ajustes obligatorios antes de ejecutar cualquier celda:

1. **Acelerador de Hardware (Accelerator)**:
   - Seleccione **GPU T4 x2** o **GPU P100**.
2. **Conexion a Internet (Internet) - OBLIGATORIO**:
   - Cambie la opcion **Internet** de *Off* a **Internet On**.
   - *Por que es necesario*: PyTorch requiere descargar los pesos preentrenados de ImageNet-1K (`resnet50` o `efficientnet_b3`) desde `download.pytorch.org`. Si Internet esta en *Off*, la celda 7 fallara con `URLError: <urlopen error [Errno -3] Temporary failure in name resolution>`.
   - *Nota*: Kaggle requiere que su cuenta este verificada por telefono para habilitar el acceso a Internet.

### Paso 4: Ajuste de Ruta en la Celda 3 del Cuaderno
En la **Celda 3** (*"2. LOCALIZADOR INTELIGENTE DEL DATASET Y DICCIONARIO DE HIPERPARAMETROS"*), ubique las lineas de configuracion iniciales:

```python
# CONFIGURACION DE RUTA KAGGLE / LOCAL / COLAB:
# Si su dataset en Kaggle tiene un nombre especifico (ej. 'dataset-anemia-oco'), puede fijarlo aqui:
# RUTA_MANUAL_KAGGLE = '/kaggle/input/dataset-anemia-oco'
# Deje en None para deteccion automatica inteligente en Kaggle, Colab o Local:
RUTA_MANUAL_KAGGLE = None
```

* **Modo Automatico (Recomendado)**:
  Deje `RUTA_MANUAL_KAGGLE = None`. El codigo explorara de forma recursiva todo `/kaggle/input`, localizando por separado la carpeta de los archivos CSV y la carpeta fisica donde se encuentran las imagenes `IMG0001.png`, soportando cualquier nivel de anidamiento.
* **Modo Manual**:
  Si prefiere fijar la ruta, copie la ruta exacta de su dataset desde el panel derecho (**Input** -> **Data** -> icono de copiar ruta) y asignela:
  ```python
  RUTA_MANUAL_KAGGLE = '/kaggle/input/dataset-anemia-oco'
  ```

### Paso 5: Ejecucion y Descarga de Resultados
1. Ejecute todas las celdas en orden secuencial (**Run All** o celda por celda con `Shift + Enter`).
2. Al finalizar el entrenamiento y evaluacion, los archivos de salida estaran disponibles en el directorio `/kaggle/working/`:
   - `best_model.pth` (pesos del mejor modelo).
   - `models_onnx/` (modelo exportado en formato ONNX para uso en aplicaciones moviles).
   - `results/` (tablas de metricas en CSV y graficas en PNG a 300 DPI).
3. Podra descargarlos directamente desde la pestana **Output** del panel derecho de Kaggle.

---

## 3. Ejecucion en Entorno Local o Google Colab

### En Computadora Local (VS Code, JupyterLab o Anaconda)
1. Abra `ResNet-50.ipynb` o `EfficientNet-B3.ipynb`.
2. El cuaderno localiza de forma automatica la carpeta `Dataset` en el sistema de archivos local (`C:\Proyectos\luu\IMAGENES\Dataset`).
3. Ejecute las celdas en orden secuencial.

### En Google Colab
1. Suba el archivo `ResNet-50.ipynb` o `EfficientNet-B3.ipynb` a Google Colab.
2. En el panel izquierdo de archivos, suba `Dataset.zip`.
3. Ejecute la Celda 0 para descomprimir el dataset:
   ```bash
   !unzip -q -o Dataset.zip
   ```
4. El localizador automatico detectara `/content/Dataset` y realizara el entrenamiento con GPU acelerada.

---

## 4. Metodologia KDD y Metricas Clinicas de la Tesis

Los cuadernos implementan el ciclo completo de Mineria de Datos:
1. **Fase 1: Seleccion y EDA Clinico**: Analisis demografico y distribucion de hemoglobina segun estandares de la OMS (< 11.0 g/dL en ninos de 6 a 59 meses).
2. **Fase 2: Preprocesamiento y Limpieza**: Fusion de transparencias RGBA a RGB sobre fondo neutro, resolucion de chunks PNG corruptos (iCCP) y validacion estricta sin fuga de pacientes.
3. **Fase 3: Transformacion y Data Augmentation**: Aumentaciones geometricas leves que preservan el color de la conjuntiva palpebral.
4. **Fase 4: Mineria de Datos y Modelado**: Transfer Learning en dos etapas (Warmup de 5 epocas y Fine-Tuning de 20 epocas con decaimiento cosenoidal y perdida balanceada por clases).
5. **Fase 5: Evaluacion Clinica e Interpretacion**: Evaluacion en el conjunto de prueba independiente (`test.csv`, n = 137), optimizacion del umbral de Youden para maximizar la Sensibilidad clinica, y generacion de curvas ROC y matrices de confusion.
6. **Fase 6: Despliegue e Interoperabilidad**: Exportacion validada a formato abierto ONNX con verificacion de paridad numerica.
