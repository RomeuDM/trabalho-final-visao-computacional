
# Projeto de Reconhecimento de Dígitos com Aprendizado de Máquina

Este projeto utiliza técnicas de aprendizado de máquina para o reconhecimento de dígitos. Composto por scripts que treinam modelos e implementam uma interface para reconhecimento de dígitos, o projeto é ideal para quem deseja entender e aplicar conceitos de classificação em imagens de dígitos, como no conjunto de dados MNIST.

### Integrantes do grupo:
    - Bernardo Zaparoli (189797)
    - Romeu Maia (190206)

## Estrutura do Projeto

O projeto é organizado da seguinte forma:

- **`modelTrain.py`**: Script responsável por treinar o modelo de reconhecimento de dígitos. Ele inclui o pré-processamento de dados, a definição do modelo e os procedimentos de treinamento e avaliação.

- **`recognize_digits.py`**: Script que carrega o modelo treinado e executa a tarefa de reconhecimento de dígitos. Esse arquivo implementa a lógica para realizar previsões com base em imagens de dígitos fornecidas como entrada.

- **`interface.py`**: Implementa a interface do usuário para o sistema de reconhecimento de dígitos. Este script configura uma interface gráfica ou API que permite ao usuário interagir com o modelo de reconhecimento de dígitos treinado.


## Pré-requisitos

Antes de executar o projeto, certifique-se de ter os seguintes pacotes instalados:

- `numpy`
- `tensorflow` ou `keras`
- `opencv-python` (se houver manipulação de imagens)

Você pode instalar esses pacotes com o comando:

```bash
pip install numpy tensorflow opencv-python
```

## Uso

1. **Treinamento do Modelo**: Execute o script `modelTrain.py` para treinar o modelo. Isso pode exigir um conjunto de dados de dígitos, como o MNIST, que será utilizado para ajustar os parâmetros do modelo.

   ```bash
   python modelTrain.py
   ```

2. **Reconhecimento de Dígitos**: Após o treinamento, utilize o script `recognize_digits.py` para testar o modelo com novas imagens de dígitos. Este script carrega o modelo salvo e realiza previsões.

   ```bash
   python recognize_digits.py
   ```

3. **Interface de Usuário**: Para interagir com o sistema através de uma interface amigável, execute o script `interface.py`. Ele fornece uma interface gráfica ou API que permite carregar imagens de dígitos e visualizar o reconhecimento realizado pelo modelo.

   ```bash
   python interface.py
   ```