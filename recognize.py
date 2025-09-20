import cv2, json, os

MODEL_PATH  = "model/lbph_model.xml"
LABELS_PATH = "model/labels.json"
CAM_INDEX   = 0
THRESHOLD   = 80 

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit("Modelo não encontrado. Rode 'python train_recognizer.py' primeiro.")
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(MODEL_PATH)
    labels = {}
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, "r", encoding="utf-8") as f:
            labels = {int(k): v for k, v in json.load(f).items()}
    return rec, labels

def main():
    recognizer, labels = load_model()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise SystemExit("Não foi possível abrir a câmera.")

    print("Pressione 'q' para sair.")
    while True:
        ok, frame = cap.read()
        if not ok: break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(80,80))

        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200,200))

            name = "Desconhecido"
            score = None
            try:
                pred_label, pred_score = recognizer.predict(roi)
                score = pred_score
                if pred_score < THRESHOLD and pred_label in labels:
                    name = labels[pred_label]
            except cv2.error:
                pass

            # retângulo + nome abaixo
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            text = name if score is None else f"{name} ({pred_score:.0f})"
            # desenha o texto logo abaixo do retângulo
            baseline = 0
            (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            # backplate
            cv2.rectangle(frame, (x, y+h), (x+tw+6, y+h+th+10), (0,0,0), -1)
            cv2.putText(frame, text, (x+3, y+h+th),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

        cv2.imshow("Golden Guard — Reconhecimento", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
