from CLM import FaceRecognitionModel
from PGP import CaptureFace

print("Treinando o modelo... Por favor, aguarde.")
model = FaceRecognitionModel(data_path=r'C:\Users\scapi\Documents\GitHub\SVM-and-PCA-Facial-Recognition\data') 

prepared_image = CaptureFace()

if prepared_image is not None:
    predictions_ranking = model.predict_and_rank(prepared_image)

    print("\n--- Ranking de Previsões ---")
    for name, probability in predictions_ranking:
        prob_percent = probability * 100
        print(f"Pessoa: {name} | Probabilidade: {prob_percent:.2f}%")

else:
    print("\nNenhuma imagem foi capturada. O programa será encerrado.")