import cv2

def detect_faces_from_webcam():
    # Carregar o classificador pré-treinado para rostos
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Capturar vídeo da webcam
    cap = cv2.VideoCapture(0)

    while True:
        # Capturar quadro a quadro
        ret, frame = cap.read()

        if not ret:
            print("Não foi possível capturar a imagem da webcam.")
            break

        # Converta a imagem capturada para escala de cinza
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detectar rostos na imagem
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Desenhe retângulos ao redor dos rostos detectados
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Exibir a imagem resultante
        cv2.imshow('Face Detection', frame)

        # Pressione 'q' para sair do loop e fechar a janela
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Libere a captura da webcam e feche todas as janelas
    cap.release()
    cv2.destroyAllWindows()

# Execute a função
detect_faces_from_webcam()
