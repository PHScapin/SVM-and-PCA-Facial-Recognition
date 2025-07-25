import numpy as np
import os
import cv2

class ImageDatasetLoader:
    def etl(data):
        """
        Extrai, transforma e carrega (ETL) imagens de um diretório para matrizes NumPy.

        Esta função varre um diretório raiz especificado. Ela assume que cada subdiretório
        dentro do diretório raiz representa uma classe (por exemplo, o nome de uma pessoa)
        e contém as imagens correspondentes a essa classe.

        O processo consiste em:
        1.  Ler cada imagem em tons de cinza.
        2.  "Achatar" (flatten) a matriz 2D de cada imagem em um vetor 1D.
        3.  Armazenar os vetores de imagem e os nomes das classes (nomes dos subdiretórios)
            em matrizes NumPy.

        A estrutura de diretório esperada é:
        
        data/
        ├── classe_1/
        │   ├── imagem1.jpg
        │   └── imagem2.png
        └── classe_2/
            ├── imagem3.jpeg
            └── imagem4.pgm

        Args:
            data (str): O caminho para o diretório raiz que contém as subpastas com as imagens.

        Returns:
            tuple[np.ndarray, np.ndarray]: Uma tupla contendo duas matrizes NumPy:
            - images_array (np.ndarray): Uma matriz onde cada **coluna** representa uma
            imagem achatada em tons de cinza. A forma da matriz é (num_pixels, num_imagens).
            - names_array (np.ndarray): Um vetor contendo os nomes das classes (rótulos)
            correspondentes a cada imagem. A forma do vetor é (num_imagens,).
        """
        
        names_array = []
        images_array = []

        with os.scandir(data) as entries:
            for entry in entries:
                with os.scandir(os.path.join(data, entry)) as images:
                    for image in images:
                        if image.is_file() and image.name.lower().endswith(('.png', '.jpg', '.jpeg', '.pgm')):
                            img = cv2.imread(image.path, cv2.IMREAD_GRAYSCALE)
                            names_array.append(entry.name)
                        
                        if img is not None:
                            flattened_img = img.flatten()
                            images_array.append(flattened_img)

        images_array = np.array(images_array)
        names_array = np.array(names_array)

        images_array = images_array
        names_array = names_array

        return images_array, names_array
