
import tensorflow as tf
from tensorflow import keras

def train_and_save_model():
    # Verificar se há GPUs disponíveis
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Configurar o TensorFlow para utilizar a GPU
            tf.config.experimental.set_memory_growth(gpus[0], True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"{len(gpus)} GPU(s) física(s) disponível(is), {len(logical_gpus)} GPU(s) lógica(s) configurada(s).")
        except RuntimeError as e:
            print(e)
    else:
        print("Nenhuma GPU encontrada. O treinamento será realizado na CPU.")

    # Carregar o conjunto de dados MNIST
    mnist = keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizar as imagens
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Redimensionar os dados para ajustar ao modelo
    x_train = x_train.reshape(-1,28,28,1)
    x_test = x_test.reshape(-1,28,28,1)

    # Construir o modelo de CNN
    model = keras.models.Sequential([
        keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2,2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    # Compilar o modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Treinar o modelo
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

    # Salvar o modelo treinado
    model.save('digit_recognition_model.h5')
    print("Modelo salvo como 'digit_recognition_model.h5'.")

if __name__ == "__main__":
    train_and_save_model()
