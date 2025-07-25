from ETL import ImageDatasetLoader
from sklearn import decomposition
from sklearn import svm
import numpy as np

class FaceRecognitionModel:
    def __init__(self, data_path):
        """
        Inicializa e treina o modelo de reconhecimento facial.

        Este construtor carrega os dados, treina o redutor de dimensionalidade (PCA)
        e o classificador (SVM). O modelo é treinado apenas uma vez.

        Args:
            data_path (str): O caminho para o diretório raiz com as imagens.
        """

        x, y = ImageDatasetLoader.etl(data_path)

        self.pca = decomposition.PCA(n_components=300, svd_solver='randomized', whiten=True).fit(x)
        x_pca = self.pca.transform(x)

        self.model = svm.SVC(C=1000, gamma=0.001, kernel='rbf', class_weight='balanced', probability=True).fit(x_pca, y)
        print("Modelo treinado com sucesso.")

    def predict_and_rank(self, image_flattened):
        """
        Prevê a probabilidade de uma imagem pertencer a cada classe e retorna um ranking.

        Args:
            image_flattened (np.ndarray): A imagem de entrada, já processada e
                                          achatada (flattened) em um vetor.
                                          Deve ter o shape (1, num_pixels).

        Returns:
            list[tuple[str, float]]: Uma lista de tuplas, onde cada tupla contém
                                     o nome da classe (pessoa) e a probabilidade
                                     associada, ordenada da maior para a menor.
        """
        # Aplica a mesma transformação PCA que foi treinada nos dados originais
        image_pca = self.pca.transform(image_flattened)

        # Obtém as probabilidades para cada classe. Retorna um array 2D, pegamos a primeira linha.
        probabilities = self.model.predict_proba(image_pca)[0]

        # Obtém os nomes das classes (ex: 'pessoa_1', 'pessoa_2') que o modelo aprendeu
        class_labels = self.model.classes_

        # Combina os nomes das classes com suas respectivas probabilidades
        results = list(zip(class_labels, probabilities))

        # Ordena a lista de resultados com base na probabilidade (o segundo item da tupla)
        # em ordem decrescente (da maior para a menor)
        ranked_results = sorted(results, key=lambda item: item[1], reverse=True)

        return ranked_results