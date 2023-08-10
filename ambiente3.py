import cv2
import numpy as np

# Carregue os arquivos YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indexes = net.getUnconnectedOutLayers()
output_layers = [layer_names[i - 1] for i in output_layers_indexes.flatten()]

# O restante do código continua o mesmo...


# Inicie a captura de vídeo
# ...

# Inicie a captura de vídeo
cap = cv2.VideoCapture(0)
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

while True:
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    # Detectando objetos
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Informações para mostrar o retângulo (classe id, confiança, coordenadas)
    class_ids = []
    confidences = []
    boxes = []

    # Para cada detecção
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Class ID 0 corresponde a pessoas
                center_x, center_y, w, h = map(int, detection[0:4] * [width, height, width, height])
                x, y = int(center_x - w / 2), int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Aplicar NMS (Non Max Suppression)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Mostrar informações na tela
    font = cv2.FONT_HERSHEY_PLAIN
    number_of_people = 0
    for i in range(len(boxes)):
        if i in indexes:
            number_of_people += 1
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y + 30), font, 2, (0, 255, 0), 2)

    cv2.putText(frame, f"Pessoas detectadas: {number_of_people}", (10, 50), font, 2, (0, 255, 0), 2)

    # Mostrar imagem
    cv2.imshow("Imagem", frame)

    # Fechar a janela pressionando a tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
