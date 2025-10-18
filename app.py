import os, cv2, json, numpy as np
from pathlib import Path

CASCADE = str(Path(__file__).resolve().parent / "haarcascade_frontalface_default.xml")
MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "lbph_model.xml"
LABELS_PATH = MODEL_DIR / "labels.json"
FACES_DIR = Path(__file__).resolve().parent / "faces"

THRESHOLD = float(os.getenv("THRESHOLD", 80))

def train_if_needed():
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        return
    print("[i] Treinando modelo (primeira execução)...")
    images, labels, names = [], [], []
    if not FACES_DIR.is_dir():
        raise SystemExit("Pasta 'faces/' não encontrada.")
    people = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])
    if not people:
        raise SystemExit("Crie subpastas em 'faces/' (uma por pessoa) com imagens.")

    face_cascade = cv2.CascadeClassifier(CASCADE)
    idx = -1
    for idx, person_dir in enumerate(people):
        names.append(person_dir.name)
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in [".jpg",".jpeg",".png",".bmp"]:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                images.append(face_roi)
                labels.append(idx)
    if not images:
        raise SystemExit("Nenhum rosto detectado em 'faces/'.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({i: n for i, n in enumerate(names)}, f, ensure_ascii=False, indent=2)
    print("[ok] Treino concluído:", names)

def load_model():
    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(str(MODEL_PATH))
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return rec, labels

def main():
    train_if_needed()
    rec, labels = load_model()
    face_cascade = cv2.CascadeClassifier(CASCADE)

    cap = cv2.VideoCapture(int(os.getenv("CAM_INDEX", 0)))
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a câmera. Ajuste CAM_INDEX.")

    from integration.simple_integration import notify_event

    print("[i] Pressione 'q' para sair.")
    last_name = None
    stable = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            roi = cv2.resize(roi, (200, 200))
            label_id, conf = rec.predict(roi)
            name = labels.get(label_id, "Desconhecido")
            color = (0, 255, 0) if conf < THRESHOLD else (0, 0, 255)

            # draw
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            text = f"{name} ({conf:.1f})"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(frame, (x, y+h), (x+tw+6, y+h+th+10), (0,0,0), -1)
            cv2.putText(frame, text, (x+3, y+h+th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

            # integration trigger with small stability
            if conf < THRESHOLD:
                if name == last_name:
                    stable += 1
                else:
                    stable = 1
                last_name = name
                if stable == 5:
                    event = notify_event(name=name, confidence=conf, bbox=[int(x), int(y), int(w), int(h)], frame_bgr=frame)
                    print("[ação] evento registrado:", event)
            else:
                stable = 0
                last_name = None

        cv2.imshow("Golden Guard — Reconhecimento", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
