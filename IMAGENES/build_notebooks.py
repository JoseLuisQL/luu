import json
import os

def create_cell(cell_type, source):
    if isinstance(source, str):
        lines = [line + '\n' for line in source.split('\n')]
        if lines and lines[-1].endswith('\n'):
            lines[-1] = lines[-1][:-1]
    else:
        lines = source
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': lines
    }
    if cell_type == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
    return cell

def save_notebook(cells, output_path):
    nb = {
        'cells': cells,
        'metadata': {
            'language_info': {
                'name': 'python',
                'version': '3.11'
            },
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"Notebook guardado exitosamente en: {output_path}")

# ==============================================================================
# 1. RESNET-50 NOTEBOOK
# ==============================================================================
resnet_cells = [
    create_cell('markdown', """# 🔬 Detección de Anemia Infantil mediante ResNet-50 y Conjuntiva Palpebral
**Tesis:** *Método no invasivo basado en redes neuronales convolucionales para la detección de la anemia infantil a partir de imágenes en el centro de salud, Ocobamba 2025*
**Institución:** Universidad Nacional José María Arguedas (UNAJMA) - Escuela Profesional de Ingeniería de Sistemas

---
### 📋 Objetivos del Cuaderno:
1. **Configuración e Hiperparámetros**: Configuración centralizada de variables (`BATCH_SIZE`, `LEARNING_RATE`, `EPOCHS`, `IMAGE_SIZE=224`).
2. **Data Pipeline & Augmentation**: Carga de imágenes de conjuntiva palpebral desde `IMAGENES/Dataset/` con aumentos biomédicos controlados.
3. **Transfer Learning con ResNet-50**: Inicialización con pesos preentrenados de ImageNet y cabezal de clasificación optimizado con Dropout y regularización.
4. **Manejo de Desbalance de Clases**: Ponderación de función de pérdida `CrossEntropyLoss` (346 Anemia vs 260 No anemia).
5. **Entrenamiento y Checkpointing**: Registro de Loss/Accuracy por época y guardado automático de `best_resnet50.pth`.
6. **Evaluación Clínica en Conjunto de Prueba (`test.csv`)**:
   - Matriz de Confusión y Curva ROC-AUC.
   - Cálculo estricto de **Sensibilidad (Recall)** y **Especificidad**.
7. **Curvas de Aprendizaje**: Gráficos listos para el Capítulo V de la tesis.
8. **Inferencia Individual**: Función `diagnosticar_anemia(image_path)` para diagnóstico clínico instantáneo."""),

    create_cell('code', """# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN DEL ENTORNO
# ==============================================================================
import os
import random
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

# Configurar estilo visual para gráficos de la tesis
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

print(f"PyTorch Version: {torch.__version__}")
print(f"TorchVision Version: {torchvision.__version__}")"""),

    create_cell('code', """# ==============================================================================
# 2. CONFIGURACIÓN CENTRALIZADA DE HIPERPARÁMETROS (MODIFICABLE)
# ==============================================================================
CONFIG = {
    # Modelo y Datos
    'MODEL_NAME': 'ResNet-50',
    'IMAGE_SIZE': 224,            # Resolución estándar de entrada para ResNet-50
    'BATCH_SIZE': 16,             # Tamaño de lote óptimo para estabilidad de gradiente
    'NUM_CLASSES': 2,             # 0: No anemia, 1: Anemia
    'CLASS_NAMES': ['No anemia', 'Anemia'],

    # Hiperparámetros de Entrenamiento
    'EPOCHS': 25,                 # Número de épocas de entrenamiento
    'LEARNING_RATE': 1e-4,        # Tasa de aprendizaje inicial
    'WEIGHT_DECAY': 1e-4,         # Regularización L2
    'DROPOUT_RATE': 0.4,          # Tasa de abandono para prevenir sobreajuste
    'SEED': 42,                   # Semilla para reproducibilidad científica

    # Rutas relativas del proyecto
    'DATA_DIR': os.path.join('..', 'IMAGENES', 'Dataset') if os.path.exists(os.path.join('..', 'IMAGENES', 'Dataset')) else os.path.join('Dataset'),
    'CHECKPOINT_PATH': 'best_resnet50.pth',
    'RESULTS_DIR': 'results'
}

# Crear carpeta de resultados si no existe
os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)

# Selección inteligente de dispositivo (CUDA / CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG['DEVICE'] = str(device)
print(f"Dispositivo de ejecución asignado: {device}")

# Fijar semilla determinista
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['SEED'])
print("Semillas fijadas para reproducibilidad.")"""),

    create_cell('code', """# ==============================================================================
# 3. VERIFICACIÓN Y CARGA DE METADATOS DEL DATASET
# ==============================================================================
dataset_base = CONFIG['DATA_DIR']
images_dir = os.path.join(dataset_base, 'images')

train_csv = os.path.join(dataset_base, 'train.csv')
val_csv = os.path.join(dataset_base, 'validation.csv')
test_csv = os.path.join(dataset_base, 'test.csv')

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

print("Distribución del Dataset (Tesis Ocobamba):")
print(f"  • Entrenamiento : {len(df_train)} imágenes -> {dict(df_train['label'].value_counts())}")
print(f"  • Validación    : {len(df_val)} imágenes -> {dict(df_val['label'].value_counts())}")
print(f"  • Prueba (Test) : {len(df_test)} imágenes -> {dict(df_test['label'].value_counts())}")
print(f"  • Total Dataset : {len(df_train) + len(df_val) + len(df_test)} imágenes registradas.")"""),

    create_cell('code', """# ==============================================================================
# 4. DATA AUGMENTATION BIOMÉDICO Y CLASE DATASET DE PYTORCH
# ==============================================================================
class AnemiaConjunctivaDataset(Dataset):
    \"\"\"Dataset personalizado para imágenes de conjuntiva palpebral y detección de anemia.\"\"\"
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label = int(row['label'])

        img_path = os.path.join(self.img_dir, img_name)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"No se encontró la imagen: {img_path}")

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

# Data Augmentation biomédico: preserva características de color clave mientras añade robustez
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Variaciones de iluminación sin alterar tinte patológico
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Crear instancias de datasets
train_dataset = AnemiaConjunctivaDataset(df_train, images_dir, transform=train_transforms)
val_dataset = AnemiaConjunctivaDataset(df_val, images_dir, transform=val_test_transforms)
test_dataset = AnemiaConjunctivaDataset(df_test, images_dir, transform=val_test_transforms)

# DataLoaders de PyTorch
train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)

print(f"DataLoaders creados exitosamente con tamaño de lote {CONFIG['BATCH_SIZE']}.")"""),

    create_cell('code', """# ==============================================================================
# 5. CONSTRUCCIÓN DE RESNET-50 CON TRANSFER LEARNING
# ==============================================================================
def build_resnet50_model(num_classes=2, dropout_rate=0.4):
    \"\"\"Construye el modelo ResNet-50 preentrenado con cabezal de clasificación personalizado.\"\"\"
    # Cargar pesos oficiales preentrenados en ImageNet
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)

    # Descongelar capas convolucionales para fine-tuning
    for param in model.parameters():
        param.requires_grad = True

    # Reemplazar la capa fully-connected (fc) final con capas densas, BatchNorm y Dropout
    in_features = model.fc.in_features # 2048 en ResNet-50
    model.fc = nn.Sequential(
        nn.Linear(in_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout_rate),
        nn.Linear(512, num_classes)
    )
    return model

model = build_resnet50_model(num_classes=CONFIG['NUM_CLASSES'], dropout_rate=CONFIG['DROPOUT_RATE']).to(device)

total_params = sum(p.num_grad() if hasattr(p, 'num_grad') else p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Modelo ResNet-50 instanciado:")
print(f"  • Parámetros Totales     : {total_params:,}")
print(f"  • Parámetros Entrenables : {trainable_params:,}")"""),

    create_cell('code', """# ==============================================================================
# 6. FUNCIÓN DE PÉRDIDA BALANCEADA, OPTIMIZADOR Y SCHEDULER
# ==============================================================================
# Calcular pesos inversos para mitigar el desbalance de clases (346 Anemia vs 260 No Anemia)
count_no_anemia = (df_train['label'] == 0).sum()
count_anemia = (df_train['label'] == 1).sum()
total_train = len(df_train)

weight_0 = total_train / (2.0 * count_no_anemia)
weight_1 = total_train / (2.0 * count_anemia)
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)

print(f"Pesos de clase calculados:")
print(f"  • Clase 0 (No anemia): {weight_0:.4f}")
print(f"  • Clase 1 (Anemia)   : {weight_1:.4f}")

# Criterio de pérdida con ponderación de clases
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Optimizador AdamW y Scheduler CosineAnnealing
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)

print("Optimizador AdamW y CosineAnnealingLR inicializados correctamente.")"""),

    create_cell('code', """# ==============================================================================
# 7. BUCLE DE ENTRENAMIENTO Y VALIDACIÓN CON CHECKPOINTING
# ==============================================================================
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_f1 = -1.0
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    start_time = time.time()
    print(f"Iniciando entrenamiento de {CONFIG['MODEL_NAME']} durante {num_epochs} épocas en {device}...\n")
    print(f"{'Época':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Val F1':<10} | {'Estado'}")
    print("-" * 85)

    for epoch in range(1, num_epochs + 1):
        # ------------------ FASE DE ENTRENAMIENTO ------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping para estabilidad
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # ------------------ FASE DE VALIDACIÓN ------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        status = ""
        # Guardar checkpoint si mejora el F1 de validación o la pérdida
        if epoch_val_f1 > best_val_f1 or (epoch_val_f1 == best_val_f1 and epoch_val_loss < best_val_loss):
            best_val_f1 = epoch_val_f1
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_f1': epoch_val_f1,
                'config': CONFIG
            }, checkpoint_path)
            status = "⭐ Mejor modelo guardado"

        print(f"{epoch:^8} | {epoch_train_loss:^12.4f} | {epoch_train_acc * 100:^11.2f}% | {epoch_val_loss:^12.4f} | {epoch_val_acc * 100:^11.2f}% | {epoch_val_f1:^10.4f} | {status}")

    elapsed = time.time() - start_time
    print(f"\nEntrenamiento finalizado en {elapsed // 60:.0f}m {elapsed % 60:.0f}s.")
    print(f"Mejor F1 en Validación: {best_val_f1:.4f} (Guardado en {checkpoint_path})")

    # Cargar los mejores pesos para la evaluación
    model.load_state_dict(best_model_weights)
    return model, history

# Ejecutar entrenamiento
model, history = train_and_validate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=CONFIG['EPOCHS'],
    checkpoint_path=CONFIG['CHECKPOINT_PATH']
)"""),

    create_cell('code', """# ==============================================================================
# 8. EVALUACIÓN CLÍNICA RIGUROSA EN EL CONJUNTO DE PRUEBA (TEST.CSV)
# ==============================================================================
model.eval()
test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.numpy())
        test_probs.extend(probs[:, 1].cpu().numpy()) # Probabilidad de Anemia (clase 1)

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_probs = np.array(test_probs)

# Matriz de Confusión
cm = confusion_matrix(test_labels, test_preds)
tn, fp, fn, tp = cm.ravel()

# Cálculo de Métricas Clínicas
acc = accuracy_score(test_labels, test_preds)
sensibilidad = recall_score(test_labels, test_preds, pos_label=1)  # TP / (TP + FN)
especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0          # TN / (TN + FP)
precision = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)

# Resumen formal para la Tesis (Capítulos V y VI)
df_metrics = pd.DataFrame({
    'Métrica Clínica / Científica': [
        'Exactitud (Accuracy)',
        'Sensibilidad (Recall - Clase Anemia)',
        'Especificidad (Tasa Verdaderos Negativos)',
        'Precisión (Valor Predictivo Positivo)',
        'F1-Score (Media Armónica)',
        'Área Bajo la Curva ROC (AUC)',
        'Verdaderos Positivos (TP)',
        'Verdaderos Negativos (TN)',
        'Falsos Positivos (FP)',
        'Falsos Negativos (FN)'
    ],
    'Valor Obtenido': [
        f"{acc * 100:.2f}%",
        f"{sensibilidad * 100:.2f}%",
        f"{especificidad * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{f1:.4f}",
        f"{roc_auc:.4f}",
        f"{tp} casos",
        f"{tn} casos",
        f"{fp} casos",
        f"{fn} casos"
    ]
})

print("=" * 65)
print("       TABLA DE RESULTADOS DE PRUEBA (TEST.CSV - 137 CASOS)")
print("=" * 65)
print(df_metrics.to_string(index=False))
print("=" * 65)

# Guardar métricas en CSV para la tesis
df_metrics.to_csv(os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_metricas_test.csv'), index=False)"""),

    create_cell('code', """# ==============================================================================
# 9. VISUALIZACIONES CLÍNICAS PARA LA TESIS (MATRIZ, ROC, CURVAS)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Curvas de Pérdida y Precisión
epochs_range = range(1, len(history['train_loss']) + 1)
axes[0].plot(epochs_range, history['train_loss'], 'b-', label='Pérdida Entrenamiento')
axes[0].plot(epochs_range, history['val_loss'], 'r--', label='Pérdida Validación')
axes[0].set_title('Evolución de Función de Pérdida', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Loss (CrossEntropy)')
axes[0].legend()
axes[0].grid(True)

# 2. Matriz de Confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1],
            xticklabels=CONFIG['CLASS_NAMES'], yticklabels=CONFIG['CLASS_NAMES'], annot_kws={'size': 14})
axes[1].set_title(f'Matriz de Confusión ({CONFIG["MODEL_NAME"]})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Diagnóstico Predicho por el Modelo', fontsize=10)
axes[1].set_ylabel('Diagnóstico Real (Gold Standard)', fontsize=10)

# 3. Curva ROC
fpr, tpr, _ = roc_curve(test_labels, test_probs)
axes[2].plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Clasificador Aleatorio')
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
axes[2].set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
axes[2].set_title('Curva Característica Operativa (ROC)', fontsize=12, fontweight='bold')
axes[2].legend(loc="lower right")
axes[2].grid(True)

plt.tight_layout()
fig_path = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_graficas_tesis.png')
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Gráficas de alta resolución guardadas en: {fig_path}")"""),

    create_cell('code', """# ==============================================================================
# 10. FUNCIÓN DE INFERENCIA CLÍNICA INDIVIDUAL
# ==============================================================================
def diagnosticar_anemia(image_path, model=model, config=CONFIG):
    \"\"\"Realiza la predicción no invasiva de anemia infantil sobre una imagen de conjuntiva.\"\"\"
    if not os.path.exists(image_path):
        print(f"Error: No existe el archivo {image_path}")
        return

    img = Image.open(image_path).convert('RGB')
    tensor = val_test_transforms(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
        pred_idx = int(np.argmax(probs))

    resultado = config['CLASS_NAMES'][pred_idx]
    prob_no_anemia = probs[0] * 100
    prob_anemia = probs[1] * 100

    print("-" * 55)
    print(f"📊 DIAGNÓSTICO PREDICTIVO - {config['MODEL_NAME']}")
    print(f"   Archivo analizado: {os.path.basename(image_path)}")
    print(f"   Resultado Clínico: >>> {resultado.upper()} <<<")
    print(f"   Probabilidad de Anemia    : {prob_anemia:.2f}%")
    print(f"   Probabilidad de No Anemia : {prob_no_anemia:.2f}%")
    print("-" * 55)
    return {'resultado': resultado, 'prob_anemia': prob_anemia, 'prob_no_anemia': prob_no_anemia}

# Prueba de inferencia con la primera imagen del conjunto de prueba
ejemplo_img = os.path.join(images_dir, df_test.iloc[0]['image'])
diagnosticar_anemia(ejemplo_img)""")
]

# ==============================================================================
# 2. EFFICIENTNET-B3 NOTEBOOK
# ==============================================================================
efficientnet_cells = [
    create_cell('markdown', """# 🔬 Detección de Anemia Infantil mediante EfficientNet-B3 y Conjuntiva Palpebral
**Tesis:** *Método no invasivo basado en redes neuronales convolucionales para la detección de la anemia infantil a partir de imágenes en el centro de salud, Ocobamba 2025*
**Institución:** Universidad Nacional José María Arguedas (UNAJMA) - Escuela Profesional de Ingeniería de Sistemas

---
### 📋 Objetivos del Cuaderno:
1. **Configuración e Hiperparámetros**: Configuración nativa para EfficientNet-B3 (`IMAGE_SIZE=300`, `BATCH_SIZE=16`, `LEARNING_RATE=2e-4`).
2. **Data Pipeline & Augmentation**: Carga de imágenes de conjuntiva palpebral desde `IMAGENES/Dataset/` con aumentos biomédicos controlados a 300x300.
3. **Transfer Learning con EfficientNet-B3**: Arquitectura basada en escalado compuesto (*Compound Scaling*), bloques MBConv (Inverted Residuals) y Squeeze-and-Excitation.
4. **Manejo de Desbalance de Clases**: Ponderación de función de pérdida `CrossEntropyLoss` (346 Anemia vs 260 No anemia).
5. **Entrenamiento y Checkpointing**: Registro de Loss/Accuracy por época y guardado de `best_efficientnet_b3.pth`.
6. **Evaluación Clínica en Conjunto de Prueba (`test.csv`)**:
   - Matriz de Confusión y Curva ROC-AUC.
   - Cálculo estricto de **Sensibilidad (Recall)** y **Especificidad**.
7. **Curvas de Aprendizaje**: Gráficos listos para el Capítulo V de la tesis.
8. **Cuadro Comparativo ResNet-50 vs EfficientNet-B3**: Síntesis automática para el Capítulo VI ("Comparación de modelos de redes neuronales")."""),

    create_cell('code', """# ==============================================================================
# 1. IMPORTACIÓN DE LIBRERÍAS Y CONFIGURACIÓN DEL ENTORNO
# ==============================================================================
import os
import random
import time
import copy
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

print(f"PyTorch Version: {torch.__version__}")
print(f"TorchVision Version: {torchvision.__version__}")"""),

    create_cell('code', """# ==============================================================================
# 2. CONFIGURACIÓN CENTRALIZADA DE HIPERPARÁMETROS (EFFICIENTNET-B3)
# ==============================================================================
CONFIG = {
    # Modelo y Datos
    'MODEL_NAME': 'EfficientNet-B3',
    'IMAGE_SIZE': 300,            # Resolución nativa óptima para EfficientNet-B3
    'BATCH_SIZE': 16,             # Tamaño de lote óptimo para estabilidad de gradiente
    'NUM_CLASSES': 2,             # 0: No anemia, 1: Anemia
    'CLASS_NAMES': ['No anemia', 'Anemia'],

    # Hiperparámetros de Entrenamiento
    'EPOCHS': 25,                 # Número de épocas de entrenamiento
    'LEARNING_RATE': 2e-4,        # Tasa de aprendizaje adaptada para EfficientNet
    'WEIGHT_DECAY': 1e-4,         # Regularización L2
    'DROPOUT_RATE': 0.3,          # Tasa de abandono estocástico
    'SEED': 42,                   # Semilla para reproducibilidad científica

    # Rutas relativas del proyecto
    'DATA_DIR': os.path.join('..', 'IMAGENES', 'Dataset') if os.path.exists(os.path.join('..', 'IMAGENES', 'Dataset')) else os.path.join('Dataset'),
    'CHECKPOINT_PATH': 'best_efficientnet_b3.pth',
    'RESULTS_DIR': 'results'
}

os.makedirs(CONFIG['RESULTS_DIR'], exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CONFIG['DEVICE'] = str(device)
print(f"Dispositivo de ejecución asignado: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

seed_everything(CONFIG['SEED'])
print("Semillas fijadas para reproducibilidad.")"""),

    create_cell('code', """# ==============================================================================
# 3. CARGA DE METADATOS Y DATA AUGMENTATION A 300x300
# ==============================================================================
dataset_base = CONFIG['DATA_DIR']
images_dir = os.path.join(dataset_base, 'images')

train_csv = os.path.join(dataset_base, 'train.csv')
val_csv = os.path.join(dataset_base, 'validation.csv')
test_csv = os.path.join(dataset_base, 'test.csv')

df_train = pd.read_csv(train_csv)
df_val = pd.read_csv(val_csv)
df_test = pd.read_csv(test_csv)

class AnemiaConjunctivaDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['image']
        label = int(row['label'])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

# Data Augmentation a resolución nativa de 300x300 para EfficientNet-B3
train_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=(-10, 10)),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_test_transforms = transforms.Compose([
    transforms.Resize((CONFIG['IMAGE_SIZE'], CONFIG['IMAGE_SIZE'])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = AnemiaConjunctivaDataset(df_train, images_dir, transform=train_transforms)
val_dataset = AnemiaConjunctivaDataset(df_val, images_dir, transform=val_test_transforms)
test_dataset = AnemiaConjunctivaDataset(df_test, images_dir, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=CONFIG['BATCH_SIZE'], shuffle=False, num_workers=0)

print(f"DataLoaders creados exitosamente para EfficientNet-B3 (resolución: {CONFIG['IMAGE_SIZE']}x{CONFIG['IMAGE_SIZE']}).")"""),

    create_cell('code', """# ==============================================================================
# 4. CONSTRUCCIÓN DE EFFICIENTNET-B3 CON TRANSFER LEARNING
# ==============================================================================
def build_efficientnet_b3_model(num_classes=2, dropout_rate=0.3):
    \"\"\"Construye el modelo EfficientNet-B3 preentrenado con cabezal clasificador personalizado.\"\"\"
    weights = models.EfficientNet_B3_Weights.DEFAULT
    model = models.efficientnet_b3(weights=weights)

    for param in model.parameters():
        param.requires_grad = True

    # Reemplazar clasificador (en EfficientNet-B3 la entrada al clasificador son 1536 características)
    in_features = model.classifier[1].in_features # 1536
    model.classifier = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, 256),
        nn.SiLU(inplace=True),
        nn.Dropout(p=0.2),
        nn.Linear(256, num_classes)
    )
    return model

model = build_efficientnet_b3_model(num_classes=CONFIG['NUM_CLASSES'], dropout_rate=CONFIG['DROPOUT_RATE']).to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Modelo EfficientNet-B3 instanciado:")
print(f"  • Parámetros Totales     : {total_params:,}")
print(f"  • Parámetros Entrenables : {trainable_params:,}")"""),

    create_cell('code', """# ==============================================================================
# 5. FUNCIÓN DE PÉRDIDA BALANCEADA Y OPTIMIZADOR
# ==============================================================================
count_no_anemia = (df_train['label'] == 0).sum()
count_anemia = (df_train['label'] == 1).sum()
total_train = len(df_train)

weight_0 = total_train / (2.0 * count_no_anemia)
weight_1 = total_train / (2.0 * count_anemia)
class_weights = torch.tensor([weight_0, weight_1], dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG['LEARNING_RATE'], weight_decay=CONFIG['WEIGHT_DECAY'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['EPOCHS'], eta_min=1e-6)

print(f"Pesos de pérdida: [No Anemia: {weight_0:.4f}, Anemia: {weight_1:.4f}]")"""),

    create_cell('code', """# ==============================================================================
# 6. BUCLE DE ENTRENAMIENTO Y VALIDACIÓN CON CHECKPOINTING
# ==============================================================================
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, checkpoint_path):
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_f1 = -1.0
    best_val_loss = float('inf')
    best_model_weights = copy.deepcopy(model.state_dict())

    start_time = time.time()
    print(f"Iniciando entrenamiento de {CONFIG['MODEL_NAME']} durante {num_epochs} épocas en {device}...\n")
    print(f"{'Época':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Val F1':<10} | {'Estado'}")
    print("-" * 85)

    for epoch in range(1, num_epochs + 1):
        # ------------------ FASE DE ENTRENAMIENTO ------------------
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels, _ in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_train += torch.sum(preds == labels.data).item()
            total_train += labels.size(0)

        scheduler.step()
        epoch_train_loss = running_loss / total_train
        epoch_train_acc = correct_train / total_train

        # ------------------ FASE DE VALIDACIÓN ------------------
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []

        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

                all_val_preds.extend(preds.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())

        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = val_correct / val_total
        epoch_val_f1 = f1_score(all_val_labels, all_val_preds, average='macro', zero_division=0)

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['val_f1'].append(epoch_val_f1)

        status = ""
        if epoch_val_f1 > best_val_f1 or (epoch_val_f1 == best_val_f1 and epoch_val_loss < best_val_loss):
            best_val_f1 = epoch_val_f1
            best_val_loss = epoch_val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'val_f1': epoch_val_f1,
                'config': CONFIG
            }, checkpoint_path)
            status = "⭐ Mejor modelo guardado"

        print(f"{epoch:^8} | {epoch_train_loss:^12.4f} | {epoch_train_acc * 100:^11.2f}% | {epoch_val_loss:^12.4f} | {epoch_val_acc * 100:^11.2f}% | {epoch_val_f1:^10.4f} | {status}")

    elapsed = time.time() - start_time
    print(f"\nEntrenamiento finalizado en {elapsed // 60:.0f}m {elapsed % 60:.0f}s.")
    print(f"Mejor F1 en Validación: {best_val_f1:.4f} (Guardado en {checkpoint_path})")

    model.load_state_dict(best_model_weights)
    return model, history

model, history = train_and_validate(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=CONFIG['EPOCHS'],
    checkpoint_path=CONFIG['CHECKPOINT_PATH']
)"""),

    create_cell('code', """# ==============================================================================
# 7. EVALUACIÓN CLÍNICA EN EL CONJUNTO DE PRUEBA (TEST.CSV)
# ==============================================================================
model.eval()
test_preds = []
test_labels = []
test_probs = []

with torch.no_grad():
    for images, labels, _ in test_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.numpy())
        test_probs.extend(probs[:, 1].cpu().numpy())

test_preds = np.array(test_preds)
test_labels = np.array(test_labels)
test_probs = np.array(test_probs)

cm = confusion_matrix(test_labels, test_preds)
tn, fp, fn, tp = cm.ravel()

acc = accuracy_score(test_labels, test_preds)
sensibilidad = recall_score(test_labels, test_preds, pos_label=1)
especificidad = tn / (tn + fp) if (tn + fp) > 0 else 0.0
precision = precision_score(test_labels, test_preds, pos_label=1, zero_division=0)
f1 = f1_score(test_labels, test_preds, pos_label=1, zero_division=0)
roc_auc = roc_auc_score(test_labels, test_probs)

df_metrics_eff = pd.DataFrame({
    'Métrica Clínica / Científica': [
        'Exactitud (Accuracy)',
        'Sensibilidad (Recall - Clase Anemia)',
        'Especificidad (Tasa Verdaderos Negativos)',
        'Precisión (Valor Predictivo Positivo)',
        'F1-Score (Media Armónica)',
        'Área Bajo la Curva ROC (AUC)',
        'Verdaderos Positivos (TP)',
        'Verdaderos Negativos (TN)',
        'Falsos Positivos (FP)',
        'Falsos Negativos (FN)'
    ],
    'Valor Obtenido': [
        f"{acc * 100:.2f}%",
        f"{sensibilidad * 100:.2f}%",
        f"{especificidad * 100:.2f}%",
        f"{precision * 100:.2f}%",
        f"{f1:.4f}",
        f"{roc_auc:.4f}",
        f"{tp} casos",
        f"{tn} casos",
        f"{fp} casos",
        f"{fn} casos"
    ]
})

print("=" * 65)
print(f"       TABLA DE RESULTADOS DE PRUEBA ({CONFIG['MODEL_NAME']})")
print("=" * 65)
print(df_metrics_eff.to_string(index=False))
print("=" * 65)

df_metrics_eff.to_csv(os.path.join(CONFIG['RESULTS_DIR'], 'efficientnet_b3_metricas_test.csv'), index=False)"""),

    create_cell('code', """# ==============================================================================
# 8. VISUALIZACIONES CLÍNICAS (PÉRDIDA, MATRIZ, ROC)
# ==============================================================================
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. Curvas de Pérdida
epochs_range = range(1, len(history['train_loss']) + 1)
axes[0].plot(epochs_range, history['train_loss'], 'g-', label='Pérdida Entrenamiento')
axes[0].plot(epochs_range, history['val_loss'], 'm--', label='Pérdida Validación')
axes[0].set_title(f'Pérdida ({CONFIG["MODEL_NAME"]})', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Épocas')
axes[0].set_ylabel('Loss (CrossEntropy)')
axes[0].legend()
axes[0].grid(True)

# 2. Matriz de Confusión
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=False, ax=axes[1],
            xticklabels=CONFIG['CLASS_NAMES'], yticklabels=CONFIG['CLASS_NAMES'], annot_kws={'size': 14})
axes[1].set_title(f'Matriz de Confusión ({CONFIG["MODEL_NAME"]})', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Diagnóstico Predicho', fontsize=10)
axes[1].set_ylabel('Diagnóstico Real', fontsize=10)

# 3. Curva ROC
fpr, tpr, _ = roc_curve(test_labels, test_probs)
axes[2].plot(fpr, tpr, color='forestgreen', lw=2, label=f'Curva ROC (AUC = {roc_auc:.3f})')
axes[2].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--', label='Clasificador Aleatorio')
axes[2].set_xlim([0.0, 1.0])
axes[2].set_ylim([0.0, 1.05])
axes[2].set_xlabel('Tasa Falsos Positivos')
axes[2].set_ylabel('Sensibilidad')
axes[2].set_title(f'Curva ROC ({CONFIG["MODEL_NAME"]})', fontsize=12, fontweight='bold')
axes[2].legend(loc="lower right")
axes[2].grid(True)

plt.tight_layout()
fig_path = os.path.join(CONFIG['RESULTS_DIR'], 'efficientnet_b3_graficas_tesis.png')
plt.savefig(fig_path, dpi=300)
plt.show()
print(f"Gráficas guardadas en: {fig_path}")"""),

    create_cell('code', """# ==============================================================================
# 9. CUADRO COMPARATIVO RESNET-50 VS EFFICIENTNET-B3 PARA CAPÍTULO VI DE LA TESIS
# ==============================================================================
resnet_csv_path = os.path.join(CONFIG['RESULTS_DIR'], 'resnet50_metricas_test.csv')
eff_csv_path = os.path.join(CONFIG['RESULTS_DIR'], 'efficientnet_b3_metricas_test.csv')

if os.path.exists(resnet_csv_path) and os.path.exists(eff_csv_path):
    df_r = pd.read_csv(resnet_csv_path).set_index('Métrica Clínica / Científica')
    df_e = pd.read_csv(eff_csv_path).set_index('Métrica Clínica / Científica')

    df_comparativa = pd.DataFrame({
        'ResNet-50': df_r['Valor Obtenido'],
        'EfficientNet-B3': df_e['Valor Obtenido']
    })

    print("=" * 70)
    print("  CUADRO COMPARATIVO PARA EL CAPÍTULO VI DE LA TESIS (UNAJMA 2026)")
    print("=" * 70)
    print(df_comparativa.to_string())
    print("=" * 70)

    comp_path = os.path.join(CONFIG['RESULTS_DIR'], 'tabla_comparativa_modelos_tesis.csv')
    df_comparativa.to_csv(comp_path)
    print(f"Tabla comparativa lista para exportar a Word/Tesis en: {comp_path}")
else:
    print("Ejecute primero ResNet-50.ipynb y luego este cuaderno para generar la tabla comparativa completa.")""")
]

# Guardar ambos notebooks
save_notebook(resnet_cells, r'C:\Proyectos\luu\IMAGENES\ResNet-50.ipynb')
save_notebook(efficientnet_cells, r'C:\Proyectos\luu\IMAGENES\EfficientNet-B3.ipynb')
