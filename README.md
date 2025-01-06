# Projeto-de-Transfer-Learning-em-Python-

# Malware Dataset Pipeline with Transfer Learning

## **1. Conectando Google Drive**
from google.colab import drive
drive.mount('/content/drive')

---

## **2. Importando Bibliotecas Necessárias**
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import zipfile

---

## **3. Extraindo o Arquivo .zip**
# Caminho do arquivo .zip no Google Drive
zip_path = '/content/drive/My Drive/malware-dataset.zip'

# Caminho onde os arquivos serão extraídos
extract_path = '/content/malware-dataset'

# Extraindo o arquivo .zip
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print(f"Arquivos extraídos para: {extract_path}")

---

## **4. Carregando Labels**
labels_path = os.path.join(extract_path, 'labels.csv')
labels = pd.read_csv(labels_path)

print(labels.head())

# Extraindo IDs e Classes
targets = labels['Class']  # Certifique-se de que 'Class' é a coluna correta
file_ids = labels['Id']

---

## **5. Convertendo Arquivos BIN para Imagens**
def bin_to_image(file_path):
    """Converte um arquivo BIN em uma imagem."""
    with open(file_path, 'rb') as file:
        binary_data = file.read()
    byte_array = np.frombuffer(binary_data, dtype=np.uint8)
    side = int(np.ceil(np.sqrt(len(byte_array))))
    padded_data = np.pad(byte_array, (0, side**2 - len(byte_array)), 'constant')
    image = padded_data.reshape(side, side)
    return image

# Caminho para os arquivos BIN
bin_dir = os.path.join(extract_path, 'BIN')

# Processa uma amostra dos arquivos
images = []
image_labels = []

for i, file_id in enumerate(file_ids[:100]):  # Ajuste o número de arquivos processados
    bin_file_path = os.path.join(bin_dir, f"{file_id}.bin")
    if os.path.exists(bin_file_path):
        img = bin_to_image(bin_file_path)
        images.append(img)
        image_labels.append(targets[i])

# Converte para arrays NumPy
images = np.array(images)
image_labels = np.array(image_labels)

print(f"Total de imagens processadas: {len(images)}")

---

## **6. Redimensionando e Normalizando Imagens**
img_size = 224

# Redimensiona imagens para um tamanho fixo
images_resized = np.array([np.resize(img, (img_size, img_size)) for img in images])

# Normaliza os pixels
images_resized = images_resized.astype('float32') / 255.0

# Codifica as labels (One-hot encoding)
num_classes = len(np.unique(image_labels))
image_labels_categorical = to_categorical(image_labels - 1, num_classes=num_classes)

---

## **7. Dividindo os Dados**
# Divisão em treino, validação e teste
x_train, x_test, y_train, y_test = train_test_split(
    images_resized, image_labels_categorical, test_size=0.2, random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, random_state=42
)

print(f"Tamanho do conjunto de treino: {x_train.shape}")
print(f"Tamanho do conjunto de validação: {x_val.shape}")
print(f"Tamanho do conjunto de teste: {x_test.shape}")

---

## **8. Construindo o Modelo com Transfer Learning**
# Carrega o modelo base (VGG16 pré-treinado no ImageNet)
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))

# Congela os pesos da VGG16
base_model.trainable = False

# Modelo personalizado
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

---

## **9. Treinando o Modelo**
# Aumentação de dados
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)
datagen.fit(x_train)

# Treinamento
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=32),
    validation_data=(x_val, y_val),
    epochs=10,
    verbose=1
)

---

## **10. Avaliando o Modelo**
# Avaliação nos dados de teste
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Acurácia no conjunto de teste: {test_accuracy:.2f}")

---

## **11. Salvando o Modelo**
model.save('/content/drive/My Drive/malware_model_vgg16.h5')

---

## **12. Visualizando os Resultados**
# Curva de treinamento
plt.plot(history.history['accuracy'], label='Acurácia - Treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia - Validação')
plt.xlabel('Épocas')
plt.ylabel('Acurácia')
plt.legend()
plt.show()
