
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import ndimage
import matplotlib.pyplot as plt

def recognize_digits_from_image(image_path):
    # Carregar o modelo treinado
    model = keras.models.load_model('digit_recognition_model.h5')

    # Ler a imagem em escala de cinza
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Verificar se a imagem foi carregada corretamente
    if img is None:
        print("Erro ao carregar a imagem. Verifique o caminho e tente novamente.")
        return

    # Pré-processamento da imagem
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret, thresh = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Remover ruídos e preencher contornos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    thresh_clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, hierarchy = cv2.findContours(thresh_clean.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma máscara em branco e desenhar contornos preenchidos
    mask = np.zeros_like(thresh_clean)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    thresh_filled = mask

    # Encontrar contornos novamente
    contours, hierarchy = cv2.findContours(thresh_filled.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ordenar contornos da esquerda para a direita
    def sort_contours(cnts):
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][0]))
        return cnts

    contours = sort_contours(contours)

    # Lista para armazenar os dígitos reconhecidos
    digits = []

    # Criar uma cópia da imagem original para desenhar os retângulos
    img_copy = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Processar cada contorno detectado
    for idx, c in enumerate(contours):
        (x, y, w, h) = cv2.boundingRect(c)

        # Ignorar pequenos contornos que podem ser ruído
        if w >= 5 and h >= 25:
            # Desenhar um retângulo ao redor do dígito detectado
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extrair o ROI (Região de Interesse) do dígito
            roi = thresh_filled[y:y + h, x:x + w]

            # Redimensionar o ROI mantendo a proporção
            if w > h:
                new_w = 20
                new_h = int(h * (20.0 / w))
            else:
                new_h = 20
                new_w = int(w * (20.0 / h))
            roi_resized = cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Criar uma imagem de 28x28 pixels e centralizar o ROI redimensionado
            roi_padded = np.zeros((28, 28), dtype=np.uint8)
            x_offset = (28 - new_w) // 2
            y_offset = (28 - new_h) // 2
            roi_padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = roi_resized

            # Centralizar o centro de massa (opcional)
            cy, cx = ndimage.center_of_mass(roi_padded)
            shiftx = np.round(14 - cx).astype(int)
            shifty = np.round(14 - cy).astype(int)
            M = np.float32([[1, 0, shiftx], [0, 1, shifty]])
            roi_padded = cv2.warpAffine(roi_padded, M, (28, 28))

            # Preparar a imagem para o modelo
            roi_normalized = roi_padded / 255.0
            roi_reshaped = roi_normalized.reshape(1,28,28,1)

            # Fazer a previsão
            prediction = model.predict(roi_reshaped)
            digit = np.argmax(prediction)

            # Adicionar o dígito à lista
            digits.append(digit)

            # Colocar o dígito previsto na imagem
            cv2.putText(img_copy, str(digit), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    # Exibir a imagem com os dígitos identificados
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB))
    plt.title('Dígitos Identificados')
    plt.axis('off')
    plt.show()

    # Compor o número completo
    number = ''.join(map(str, digits))
    print('Número reconhecido:', number)
    return number

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Uso: python recognize_digits.py caminho_da_imagem")
    else:
        image_path = sys.argv[1]
        recognize_digits_from_image(image_path)
