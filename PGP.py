import cv2
import numpy as np

def CaptureFace():
    """
    Captura uma imagem da webcam, detecta um rosto, processa a imagem
    e a retorna como um vetor unidimensional (flatten).

    O processo consiste em:
    1.  Abrir a webcam e exibir o feed de vídeo.
    2.  Ao pressionar a tecla 's', captura o frame atual.
    3.  Utiliza o classificador Haar Cascade para detectar o rosto no frame.
    4.  Recorta a região do rosto detectado.
    5.  Converte a imagem do rosto para escala de cinza.
    6.  Redimensiona a imagem para 94x125 pixels.
    7.  "Achata" (flatten) a matriz da imagem em um vetor 1D.

    Certifique-se de que o arquivo 'haarcascade_frontalface_default.xml' está
    no mesmo diretório ou forneça o caminho completo para ele.

    Returns:
        np.ndarray: Um vetor NumPy unidimensional representando a imagem do
        rosto processada, ou None se nenhum rosto for detectado ou a
        captura for cancelada.
    """
        
    face_cascade = cv2.CascadeClassifier('env\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')

    # Inicia a captura de vídeo da webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Erro: Não foi possível abrir a webcam.")
        return None

    print("\nPressione 's' para salvar a foto ou 'q' para sair.")

    while True:
        # Captura frame por frame
        ret, frame = cap.read()
        if not ret:
            print("Erro: Não foi possível capturar o frame.")
            break

        # Exibe o resultado
        cv2.imshow('Webcam - Pressione "s" para capturar ou "q" para sair', frame)

        key = cv2.waitKey(1) & 0xFF

        # Se a tecla 's' for pressionada, captura a imagem
        if key == ord('s'):
            # Converte o frame capturado para escala de cinza para a detecção
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detecta faces no frame
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            if len(faces) > 0:
                # Pega as coordenadas da primeira face detectada
                (x, y, w, h) = faces[0]

                # Recorta a imagem para obter apenas o rosto
                face_image = frame[y:y+h, x:x+w]
                
                # --- Início do Processamento similar ao ETL ---

                # 1. Converte para escala de cinza
                gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

                # 2. Redimensiona a imagem para o formato desejado (125 de altura, 94 de largura)
                resized_face = cv2.resize(gray_face, (94, 125))

                # 3. "Achata" (flatten) a imagem em um vetor 1D
                flattened_face = resized_face.flatten()
                
                print("\nFoto capturada e processada com sucesso!")
                
                # Libera a webcam e fecha as janelas
                cap.release()
                cv2.destroyAllWindows()

                return flattened_face.reshape(1, -1) # Retorna no formato (1, n_features) para o predict
            
            else:
                print("\nNenhum rosto detectado. Tente novamente.")

        # Se a tecla 'q' for pressionada, sai do loop
        elif key == ord('q'):
            print("\nCaptura cancelada.")
            break

    # Libera a webcam e fecha as janelas
    cap.release()
    cv2.destroyAllWindows()
    return None


