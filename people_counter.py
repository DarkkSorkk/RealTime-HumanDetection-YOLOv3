import cv2

def detect_people(frame):
    # Inicializa o detector de pessoas para câmeras padrão
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    # Detecta pessoas na imagem
    rects, weights = hog.detectMultiScale(frame, winStride=(4, 4), padding=(8, 8), scale=1.05)

    return rects

def main():
    cap = cv2.VideoCapture(0)  # Captura vídeo da webcam

    while True:
        ret, frame = cap.read()  # Lê um quadro do vídeo

        if not ret:
            break

        detected_people = detect_people(frame)

        for (x, y, w, h) in detected_people:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Desenha retângulos em torno das pessoas detectadas

        # Exibe o quadro com as detecções e a contagem de pessoas
        cv2.putText(frame, f"Number of people: {len(detected_people)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow("People Count", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Pressione 'q' para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
