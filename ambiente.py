import cv2

# Inicializa o detector de pessoas com o padrão HOG
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Captura o vídeo da webcam
cap = cv2.VideoCapture(0)

while True:
    # Lê um frame do vídeo
    ret, frame = cap.read()

    # Redimensiona o frame para acelerar o processo de detecção
    frame = cv2.resize(frame, (640, 480))

    # Detecta as pessoas no frame
    boxes, weights = hog.detectMultiScale(frame, winStride=(8, 8))

    # Desenha retângulos em volta das pessoas detectadas
    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Exibe o frame com as detecções
    cv2.imshow('Pessoas Detectadas', frame)

    # Sai do loop se a tecla 'q' for pressionada
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera o objeto de captura de vídeo e fecha as janelas
cap.release()
cv2.destroyAllWindows()
