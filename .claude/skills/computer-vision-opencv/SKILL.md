---
name: computer-vision-opencv
description: Visión por computadora avanzada, procesamiento digital de imágenes, Data Augmentation biomédico y visión profunda con PyTorch y OpenCV (origen skills.sh / mindrally).
---

# Computer Vision & Image Processing (skills.sh Standard)

## Principios Fundamentales
1. **Pipelines Funcionales de Imagen**:
   - Representación uniforme de canales (RGB para PyTorch `[C, H, W]` normalizado en `[0.0, 1.0]`).
   - Transformaciones funcionales reproducibles con `torchvision.transforms.v2` o `torchvision.transforms`.
   - Normalización estándar según la distribución del preentrenamiento (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`).

2. **Aumento de Datos (Data Augmentation) Biomédico**:
   - En imágenes clínicas de conjuntiva palpebral o tejidos biológicos, aplicar aumentos que simulen variabilidad real sin alterar la patología:
     - Volteos horizontales (`RandomHorizontalFlip`).
     - Rotaciones angulares leves (`RandomRotation(degrees=(-10, 10))`).
     - Ajustes ligeros de brillo y contraste (`ColorJitter(brightness=0.1, contrast=0.1)`), evitando cambios drásticos de matiz (hue) que falseen la palidez característica de la anemia.
     - Redimensionado bicúbico o bilineal conservando la relación de aspecto o con recorte aleatorio suave (`RandomResizedCrop(scale=(0.85, 1.0))`).

3. **Inferencia y Post-Procesamiento**:
   - Carga de imágenes con `PIL.Image.open(path).convert('RGB')`.
   - Desactivación estricta de gradientes en inferencia: `with torch.no_grad():`.
   - Aplicación de `torch.softmax(logits, dim=1)` para obtener probabilidades calibradas por clase.
