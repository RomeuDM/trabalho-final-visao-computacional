
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from recognize_digits import recognize_digits_from_image
import os

class PaintApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconhecimento de Dígitos Manuscritos")
        self.root.resizable(False, False)
        
        self.tela_tamanho = 400  # tamanho em pixels
        self.brush_color = 'black'
        self.brush_size = 8

        # Configurar Canvas
        self.canvas = tk.Canvas(self.root, width=self.tela_tamanho, height=self.tela_tamanho, bg='white', cursor="cross")
        self.canvas.grid(row=0, column=0, columnspan=4)

        # Criar uma imagem PIL em branco e objeto de desenho
        self.image = Image.new("RGB", (self.tela_tamanho, self.tela_tamanho), "white")
        self.draw = ImageDraw.Draw(self.image)

        # Vincular eventos do mouse
        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_move_press)

        # Botões de controle
        self.clear_button = tk.Button(self.root, text="Limpar Tela", command=self.clear_canvas)
        self.clear_button.grid(row=1, column=0, pady=5)

        self.recognize_button = tk.Button(self.root, text="Reconhecer Dígitos", command=self.recognize_digits)
        self.recognize_button.grid(row=1, column=1, pady=5)

        self.color_button = tk.Button(self.root, text="Cor do Pincel", command=self.choose_color)
        self.color_button.grid(row=1, column=2, pady=5)

        self.brush_size_button = tk.Button(self.root, text="Tamanho do Pincel", command=self.change_brush_size)
        self.brush_size_button.grid(row=1, column=3, pady=5)

        # Área de resultado
        self.result_label = tk.Label(self.root, text="Resultado: ", font=("Helvetica", 16))
        self.result_label.grid(row=2, column=0, columnspan=4, pady=10)

    def on_button_press(self, event):
        # Inicia o desenho
        self.last_x = event.x
        self.last_y = event.y

    def on_move_press(self, event):
        # Desenha no Canvas
        self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                fill=self.brush_color, width=self.brush_size, capstyle=tk.ROUND, smooth=True)

        # Desenha na imagem PIL
        self.draw.line([self.last_x, self.last_y, event.x, event.y],
                       fill=self.brush_color, width=self.brush_size)

        # Atualiza as coordenadas
        self.last_x = event.x
        self.last_y = event.y

    def clear_canvas(self):
        # Limpa o Canvas e a imagem PIL
        self.canvas.delete("all")
        self.image = Image.new("RGB", (self.tela_tamanho, self.tela_tamanho), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.result_label.config(text="Resultado: ")

    def recognize_digits(self):
        # Salvar a imagem desenhada temporariamente
        temp_image_path = "temp_digit_image.png"
        self.image.save(temp_image_path)

        # Chamar a função de reconhecimento
        number = recognize_digits_from_image(temp_image_path)

        # Atualizar o resultado na interface
        if number:
            self.result_label.config(text=f"Resultado: {number}")
        else:
            self.result_label.config(text="Nenhum dígito reconhecido.")

        # Remover a imagem temporária
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

    def choose_color(self):
        # Abre o seletor de cores
        color_code = tk.colorchooser.askcolor(title="Escolha a cor do pincel")
        if color_code[1]:
            self.brush_color = color_code[1]

    def change_brush_size(self):
        # Janela para mudar o tamanho do pincel
        size_window = tk.Toplevel(self.root)
        size_window.title("Tamanho do Pincel")
        size_window.resizable(False, False)

        tk.Label(size_window, text="Escolha o tamanho do pincel (1-50):").pack(pady=5)
        size_var = tk.IntVar(value=self.brush_size)
        size_entry = tk.Entry(size_window, textvariable=size_var)
        size_entry.pack(pady=5)

        def apply_size():
            size = size_var.get()
            if 1 <= size <= 50:
                self.brush_size = size
                size_window.destroy()
            else:
                tk.messagebox.showerror("Erro", "O tamanho deve ser entre 1 e 50.")

        tk.Button(size_window, text="Aplicar", command=apply_size).pack(pady=5)

if __name__ == "__main__":
    root = tk.Tk()
    app = PaintApp(root)
    root.mainloop()
