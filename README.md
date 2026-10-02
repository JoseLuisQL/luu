# Sistema No Invasivo de Deteccion de Anemia Infantil mediante Redes Neuronales Convolucionales

Proyecto de investigacion aplicada y desarrollo computacional para la tesis:
> **"METODO NO INVASIVO BASADO EN REDES NEURONALES CONVOLUCIONALES PARA LA DETECCION DE LA ANEMIA INFANTIL A PARTIR DE IMAGENES EN EL CENTRO DE SALUD, OCOBAMBA 2025"**  
> *Universidad Nacional Jose Maria Arguedas (UNAJMA) - 2026*

---

## 1. Resumen del Sistema

El sistema implementa un metodo computacional no invasivo para la clasificacion y tamizaje oportuno de anemia infantil en ninos de 6 a 59 meses de edad, evaluando imagenes digitales de la conjuntiva palpebral anterior. Utiliza arquitecturas de aprendizaje profundo (Deep Learning) basadas en Redes Neuronales Convolucionales (CNN) con aprendizaje por transferencia (Transfer Learning), eliminando la necesidad de puncion venosa o capilar para el tamizaje preliminar en centros de salud rurales.

### Componentes y Metodologia KDD

El flujo del sistema se estructura en seis fases bajo la metodologia KDD (Knowledge Discovery in Databases):

1. **Seleccion y Analisis Exploratorio Clinico (EDA)**:
   - Particionamiento estratificado disjunto a nivel de paciente (`patient_id`) para evitar cualquier fuga de datos (*data leakage*).
   - Analisis de concentracion serica de hemoglobina (g/dL) bajo el estandar diagnostico de la Organizacion Mundial de la Salud (OMS), fijando anemia en valores menores a 11.0 g/dL.
2. **Preprocesamiento y Limpieza de Imagenes**:
   - Algoritmo de lectura robusto con tolerancia a fallas de metadatos (reparacion de errores CRC en chunks iCCP de archivos PNG como `T_A95_img_002.png`).
   - Conversion automatica de imagenes con canal alfa (RGBA) mediante composicion sobre fondo blanco neutro a formato RGB tricanal estandar.
   - Fallback hibrido con OpenCV ante excepciones de la libreria PIL.
3. **Transformacion y Aumento de Datos (Data Augmentation)**:
   - Normalizacion parametrica basada en ImageNet (medias y desviaciones estandar por canal).
   - Modificaciones geometricas y fotometricas moderadas (rotaciones de +/- 15 grados, giros horizontales, ligeras variaciones de brillo/contraste de +/- 10%) que preservan la palidez vascular y el matiz cromatico de la mucosa palpebral.
   - Resolucion espacial adaptada: $224 \times 224$ pixeles para ResNet-50 y $300 \times 300$ pixeles para EfficientNet-B3.
4. **Mineria de Datos y Modelado en Dos Fases**:
   - **ResNet-50**: Arquitectura residual profunda de 24.5 millones de parametros con conexiones de salto (*residual skip connections*) que mitigan el desvanecimiento del gradiente.
   - **EfficientNet-B3**: Arquitectura optimizada de 12.2 millones de parametros con convoluciones MBConv (Mobile Inverted Bottleneck Conv) y mecanismos de atencion Squeeze-and-Excitation (SE).
   - **Estrategia de entrenamiento en dos etapas**:
     * *Fase 1 (Calentamiento / Warmup - 5 epocas)*: Congelamiento del extractor de caracteristicas base (*backbone*) y optimizacion de la cabeza de clasificacion con tasa de aprendizaje mayor ($10^{-3}$).
     * *Fase 2 (Ajuste Fino / Fine-Tuning - 20 epocas)*: Descongelamiento gradual del extractor con tasa de aprendizaje reducida ($5 \times 10^{-5}$ a $10^{-4}$), decaimiento cosenoidal (`CosineAnnealingLR`) y regularizacion de pesos (`weight decay` de $10^{-2}$).
   - **Ponderacion de perdida**: Funcion de perdida de entropia cruzada con pesos inversos a la frecuencia de clase para contrarrestar el desbalance de casos positivos y controles normales.
5. **Evaluacion Clinica y Calibracion de Umbral**:
   - Evaluacion final en conjunto de prueba independiente ciego (`test.csv`, $n = 137$, 50.4% anemia vs. 49.6% no anemia).
   - Maximizacion del indice de Youden ($J = \text{Sensibilidad} + \text{Especificidad} - 1$) sobre el conjunto de validacion para determinar el umbral de clasificacion optimo, priorizando la reduccion de falsos negativos clinicos.
   - Estimacion de intervalos de confianza al 95% (Wilson / Clopper-Pearson) para Sensibilidad, Especificidad, Precision, Valor Predictivo Positivo (VPP), Valor Predictivo Negativo (VPN), F1-Score y Area Bajo la Curva ROC (AUC).
6. **Exportacion e Interoperabilidad**:
   - Serializacion a formato abierto ONNX (`.onnx`) con verificacion de paridad numerica frente al modelo PyTorch ($\text{Error} < 10^{-4}$) para su integracion en dispositivos moviles de campo.
   - Generacion de checkpoints de pesos optimos (`best_model.pth`).

---

## 2. Estructura y Llaves del Dataset

El conjunto de datos comprende 888 registros medicos y fotograficos distribuidos en cuatro archivos tabulares y una carpeta de imagenes:

```text
Dataset/
├── images/             # 888 imagenes en formatos PNG y JPG (IMG0001.png a IMG0888.png)
├── metadata.csv        # Registro maestro general (888 filas, 10 columnas)
├── train.csv           # Particion de entrenamiento (606 filas, 68.24%)
├── validation.csv      # Particion de validacion (145 filas, 16.33%)
└── test.csv            # Particion de prueba independiente (137 filas, 15.43%)
```

### Definicion de Columnas y Llaves del Dataset

| Llave / Columna | Tipo de Dato | Rol en el Sistema | Proposito Clinico y Computacional |
| :--- | :--- | :--- | :--- |
| **`image`** | Texto (`str`) | **Llave Primaria (PK)** | Nombre exacto del archivo fisico (ej. `IMG0001.png`). Es 100% unica por fila y enlaza el registro tabular con el tensor de imagen. |
| **`patient_id`** | Texto (`str`) | **Llave Foranea (FK) / Agrupador** | Identificador del paciente (761 pacientes unicos para 888 imagenes). Asegura que tomas multiples del mismo nino no se compartan entre entrenamiento y prueba. |
| **`label`** | Entero (`int64`) | **Variable Objetivo (*Target*)** | Etiqueta binaria de clasificacion: `1` = Diagnostico positivo de Anemia, `0` = No anemia (control). |
| **`Resultado_anemia`** | Texto (`str`) | **Descriptor Categorico** | Representacion textual de la condicion clinica (`Anemia` / `No anemia`) para reportes y visualizaciones. |
| **`HB`** | Decimal (`float64`) | **Biomarcador Cuantitativo** | Concentracion serica de hemoglobina en sangre (g/dL). Estandar OMS: anemia si es menor a 11.0 g/dL en ninos de 6 a 59 meses. |
| **`HB_Nivel`** | Decimal (`float64`) | **Variable Redundante** | Duplicado de la columna `HB` preservado por compatibilidad historica. |
| **`Edad_meses`** | Decimal (`float64`) | **Variable Demografica** | Edad cronologica en meses cumplidos al momento de la evaluacion (rango: 6 a 60 meses). |
| **`Sexo`** | Texto (`str`) | **Variable Demografica** | Sexo biologico del infante (`Male` / `Female`). |
| **`source`** | Texto (`str`) | **Trazabilidad Institucional** | Procedencia de la toma (`anemia_dataset` o `data_tes`). |
| **`mask`** | Nulo (`float64`) | **Segmentacion Espacial** | Campo reservado para mascaras de segmentacion palpebral. |

---

## 3. Guia Paso a Paso: Como Subir el Dataset a Kaggle Correctamente

Para que Kaggle monte el dataset con una jerarquia limpia y sin carpetas redundantes que compliquen la busqueda de archivos, siga estas instrucciones:

### Paso 1: Empaquetar los Archivos en su Computadora
1. Abra el explorador de archivos en la carpeta:
   `C:\Proyectos\luu\IMAGENES\Dataset\`
2. Seleccione simultaneamente:
   - La carpeta `images`
   - Los 4 archivos: `metadata.csv`, `train.csv`, `validation.csv`, `test.csv`
3. Haga clic derecho y elija **Comprimir en archivo ZIP** (o use 7-Zip / WinRAR).
4. Asigne un nombre al archivo comprimido, por ejemplo: `dataset-anemia-ocobamba.zip`.

*Nota de estructura interna*: Al abrir el archivo `.zip`, en la raiz deben verse directamente la carpeta `images/` y los cuatro archivos `.csv`, sin una carpeta padre contenedora adicional.

### Paso 2: Crear el Dataset en la Plataforma Web de Kaggle
1. Inicie sesion en [kaggle.com](https://www.kaggle.com).
2. En la barra de navegacion lateral izquierda, haga clic en **Datasets**.
3. Haga clic en el boton **+ New Dataset** (ubicado en la esquina superior derecha).
4. En el campo **Dataset Title**, escriba un nombre representativo, por ejemplo:
   `dataset-anemia-oco` (o `dataset-anemia-ocobamba`).
5. Arrastre el archivo `dataset-anemia-ocobamba.zip` hacia la ventana de carga.
6. Espere a que la barra de progreso finalice la carga y el procesamiento.
7. Haga clic en el boton **Create** (Crear).
8. Una vez creado, Kaggle asignara un punto de montaje en el contenedor:
   `/kaggle/input/<nombre-del-dataset>/`
   (Por ejemplo: `/kaggle/input/dataset-anemia-oco/`).

---

## 4. Guia Paso a Paso: Como Configurar y Ejecutar los Cuadernos en Kaggle

### Paso 1: Importar el Cuaderno a Kaggle
1. En Kaggle, dirijase a **Code** -> **+ New Notebook**.
2. En la barra de herramientas del cuaderno, seleccione **File** -> **Import Notebook**.
3. Seleccione el archivo desde su computadora:
   - Para ResNet-50: `C:\Proyectos\luu\IMAGENES\prueba-anemia-resnet.ipynb` (o `ResNet-50.ipynb`)
   - Para EfficientNet-B3: `C:\Proyectos\luu\IMAGENES\prueba-anemia-eficcient.ipynb` (o `EfficientNet-B3.ipynb`)
4. El cuaderno se cargara en el entorno interactivo de Kaggle.

### Paso 2: Vincular el Dataset al Cuaderno
1. En el panel lateral derecho del cuaderno, localice la seccion **Input**.
2. Haga clic en **+ Add Data**.
3. Seleccione la pestana **Your Datasets** (Tus conjuntos de datos).
4. Busque el dataset subido en el paso anterior (`dataset-anemia-oco`) y haga clic en el boton **+** para agregarlo.
5. Verifique que ahora en la seccion **Input** -> **Data** aparezca listado su dataset con sus subcarpetas.

### Paso 3: Ajustes Criticos de la Sesion en Kaggle (Panel Derecho "Notebook options")
En la barra lateral derecha del cuaderno, configure las siguientes opciones obligatorias:

1. **Acelerador de Hardware (Accelerator)**:
   - Cambie de *None* a **GPU T4 x2** (o **GPU P100**).
   - Esto acelera el entrenamiento mediante CUDA y PyTorch en GPU.
2. **Conexion a Internet (Internet) - CONFIGURACION OBLIGATORIA**:
   - Ubique la opcion **Internet** y active el interruptor a **Internet On** (Internet connected).
   - *Motivo fundamental*: PyTorch necesita descargar los pesos oficiales preentrenados de ImageNet-1K (`resnet50` o `efficientnet_b3`) desde `download.pytorch.org`. Si esta opcion permanece en *Internet Off*, la ejecucion fallara con el error `URLError: <urlopen error [Errno -3] Temporary failure in name resolution>`.
   - *Requisito de Kaggle*: Kaggle solicita que la cuenta de usuario este verificada mediante numero telefonico (SMS) para permitir el encendido de conexion a Internet.

### Paso 4: Verificacion y Configuracion de Rutas en la Celda 3 del Cuaderno
Dirijase a la **Celda 3** del cuaderno (titulada: `2. LOCALIZADOR INTELIGENTE DEL DATASET Y DICCIONARIO DE HIPERPARAMETROS`).

En las primeras lineas de codigo encontrara la variable de configuracion:

```python
# CONFIGURACION DE RUTA KAGGLE / LOCAL / COLAB:
# Si su dataset en Kaggle tiene un nombre especifico (ej. 'dataset-anemia-oco'), puede fijarlo aqui:
# RUTA_MANUAL_KAGGLE = '/kaggle/input/dataset-anemia-oco'
# Deje en None para deteccion automatica inteligente en Kaggle, Colab o Local:
RUTA_MANUAL_KAGGLE = None
```

Tiene dos modalidades de ejecucion:

* **Modalidad Recomendada (Deteccion Automatica - Mantener en `None`)**:
  Deje `RUTA_MANUAL_KAGGLE = None`. El algoritmo `resolver_rutas_dataset()` inspeccionara recursivamente `/kaggle/input`, localizando de forma desacoplada la carpeta donde estan los 4 archivos CSV y la carpeta donde estan las imagenes `IMG0001.png`, resolviendo automaticamente cualquier nivel de subcarpetas creado por Kaggle.
* **Modalidad Manual (Especificacion Directa de Ruta)**:
  Si desea fijar la ruta de forma explicita:
  1. En el panel lateral derecho, en **Input** -> **Data**, pase el cursor sobre su dataset y haga clic en el icono de copiar ruta (*Copy path*).
  2. Asigne esa ruta a la variable:
     ```python
     RUTA_MANUAL_KAGGLE = '/kaggle/input/dataset-anemia-oco'
     ```
     (Si al descomprimir se creo una subcarpeta interna llamada `Dataset`, coloque: `/kaggle/input/dataset-anemia-oco/Dataset`).

### Paso 5: Ejecucion del Cuaderno y Obtencion de Resultados
1. En la barra superior, seleccione **Run** -> **Run All** (o ejecute cada celda secuencialmente de arriba hacia abajo con `Shift + Enter`).
2. El sistema completara:
   - Validacion y auditoria de las 888 imagenes.
   - Creacion de los DataLoaders estratificados.
   - Entrenamiento Fase 1 (Warmup de 5 epocas).
   - Entrenamiento Fase 2 (Fine-Tuning de 20 epocas con decaimiento cosenoidal).
   - Evaluacion en el conjunto de prueba independiente (`test.csv`, $n = 137$).
   - Calibracion del umbral optimo de Youden.
   - Generacion de matrices de confusion, curvas ROC-AUC y graficas a 300 DPI.
   - Exportacion del modelo final a formato ONNX.
3. **Descarga de archivos generados**:
   - Todos los resultados se guardan en el directorio de trabajo `/kaggle/working/`.
   - En el panel lateral derecho, en la seccion **Output**, encontrara los archivos listos para descargar:
     * `best_model.pth` (pesos del mejor modelo segun perdida de validacion).
     * `models_onnx/` (modelo serializado en formato interoperable ONNX).
     * `results/` (tablas de metricas en formato CSV y graficas de resolucion academica en formato PNG).

---

## 5. Metricas Clinicas Registradas para la Tesis

| Metrica | Formula Matematica | Criterio Clinico en Salud Infantil |
| :--- | :--- | :--- |
| **Sensibilidad (*Recall*)** | $\frac{TP}{TP + FN}$ | **Prioridad Maxima**: Evita que ninos con anemia sean falsamente diagnosticados como sanos (falsos negativos). |
| **Especificidad** | $\frac{TN}{TN + FP}$ | Evita tratamientos de suplementacion de hierro innecesarios en ninos con niveles normales de hemoglobina. |
| **Exactitud (*Accuracy*)** | $\frac{TP + TN}{\text{Total}}$ | Porcentaje global de predicciones correctas sobre la muestra de prueba. |
| **Precision (*VPP*)** | $\frac{TP}{TP + FP}$ | Certeza de que un paciente clasificado como positivo realmente padece anemia. |
| **F1-Score** | $2 \cdot \frac{\text{Precision} \cdot \text{Sensibilidad}}{\text{Precision} + \text{Sensibilidad}}$ | Media armonica entre precision y sensibilidad ante muestras con leve desbalance. |
| **Area Bajo la Curva (ROC-AUC)** | $\int_{0}^{1} \text{TPR}(\text{FPR}) \, d\text{FPR}$ | Capacidad global de discriminacion del clasificador independiente del umbral operativo. |

---

## 6. Ejecucion Alternativa: Entorno Local o Google Colab

* **En Computadora Local (VS Code / JupyterLab)**:
  Los cuadernos detectan de forma nativa la carpeta `C:\Proyectos\luu\IMAGENES\Dataset`. Simplemente abra `ResNet-50.ipynb` o `EfficientNet-B3.ipynb` y ejecute todas las celdas.
* **En Google Colab**:
  Suba el cuaderno respectivo y el archivo comprimido `Dataset.zip` al panel de archivos de Colab (`/content/`). La Celda 0 o el resolutor automatico detectaran `/content/Dataset` y procedera con el entrenamiento en GPU acelerada (T4).
