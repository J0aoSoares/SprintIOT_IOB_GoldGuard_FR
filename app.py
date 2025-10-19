import argparse
import logging
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

BASE_DIR = Path(__file__).resolve().parent
CASCADE = BASE_DIR / "haarcascade_frontalface_default.xml"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "lbph_model.xml"
LABELS_PATH = MODEL_DIR / "labels.json"
FACES_DIR = BASE_DIR / "faces"

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp"}

def configure_logging():
    logging.basicConfig(format="[%(levelname)s] %(message)s", level=logging.INFO)

def ensure_opencv_face_module():
    if not hasattr(cv2, "face") or not hasattr(cv2.face, "LBPHFaceRecognizer_create"):
        raise SystemExit("OpenCV build has no 'cv2.face' module. Install opencv-contrib-python.")

def train_if_needed(threshold: float = 80.0) -> None:
    """Train LBPH model from images in `faces/` if model files don't exist."""
    if MODEL_PATH.exists() and LABELS_PATH.exists():
        logging.info("Modelo já treinado encontrado. Pulando treino.")
        return

    logging.info("Treinando modelo (primeira execução)...")
    if not FACES_DIR.is_dir():
        raise SystemExit("Pasta 'faces/' não encontrada.")

    people = sorted([d for d in FACES_DIR.iterdir() if d.is_dir()])
    if not people:
        raise SystemExit("Crie subpastas em 'faces/' (uma por pessoa) com imagens.")

    face_cascade = cv2.CascadeClassifier(str(CASCADE))
    images, labels, names = [], [], []
    for idx, person_dir in enumerate(people):
        names.append(person_dir.name)
        for img_path in person_dir.glob("*"):
            if img_path.suffix.lower() not in ALLOWED_EXT:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))
            for (x, y, w, h) in faces:
                face_roi = gray[y:y+h, x:x+w]
                face_roi = cv2.resize(face_roi, (200, 200))
                images.append(face_roi)
                labels.append(idx)

    if not images:
        raise SystemExit("Nenhum rosto detectado em 'faces/'.")

    ensure_opencv_face_module()
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, np.array(labels))

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    recognizer.write(str(MODEL_PATH))
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump({i: n for i, n in enumerate(names)}, f, ensure_ascii=False, indent=2)

    logging.info("Treino concluído: %s", names)


def load_model() -> Tuple[object, Dict[int, str]]:
    ensure_opencv_face_module()
    rec = cv2.face.LBPHFaceRecognizer_create()
    if not MODEL_PATH.exists():
        raise SystemExit("Modelo não encontrado. Rode o programa com imagens em 'faces/' para treinar.")
    rec.read(str(MODEL_PATH))
    if not LABELS_PATH.exists():
        raise SystemExit("Arquivo de labels não encontrado (labels.json).")
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    return rec, labels


def draw_label(frame, x, y, w, h, text, color):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.rectangle(frame, (x, y + h), (x + tw + 6, y + h + th + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (x + 3, y + h + th), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)


def main():
    configure_logging()
    parser = argparse.ArgumentParser(description="Golden Guard — Reconhecimento facial simples")
    parser.add_argument("--cam", type=int, default=int(os.getenv("CAM_INDEX", 0)), help="Índice da câmera")
    parser.add_argument("--threshold", type=float, default=float(os.getenv("THRESHOLD", 80)), help="Limite de confiança (menor é melhor)")
    parser.add_argument("--stability", type=int, default=int(os.getenv("STABILITY", 5)), help="Quadros consecutivos para considerar estável")
    parser.add_argument("--min-size", type=int, default=int(os.getenv("MIN_SIZE", 80)), help="Tamanho mínimo do rosto detectado")
    parser.add_argument("--no-gui", action="store_true", help="Não exibir janela (útil em servidores)")
    args = parser.parse_args()

    train_if_needed()
    rec, labels = load_model()
    face_cascade = cv2.CascadeClassifier(str(CASCADE))

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise SystemExit("Não consegui abrir a câmera. Ajuste CAM_INDEX/--cam.")

    # lazy import of integration to avoid startup errors when not needed
    try:
        from integration.simple_integration import notify_event
    except Exception:  # keep program working even if integration fails
        def notify_event(**kwargs):
            logging.debug("notify_event chamado, mas integração não disponível: %s", kwargs)
            return None

    logging.info("Pressione 'q' para sair (se GUI estiver ativada).")
    last_name = None
    stable = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("Falha ao ler frame da câmera. Saindo.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize=(args.min_size, args.min_size))

            for (x, y, w, h) in faces:
                roi = gray[y:y + h, x:x + w]
                roi = cv2.resize(roi, (200, 200))
                label_id, conf = rec.predict(roi)
                name = labels.get(label_id, "Desconhecido")
                color = (0, 255, 0) if conf < args.threshold else (0, 0, 255)

                draw_label(frame, x, y, w, h, f"{name} ({conf:.1f})", color)

                # integration trigger with small stability
                if conf < args.threshold:
                    if name == last_name:
                        stable += 1
                    else:
                        stable = 1
                    last_name = name
                    if stable >= args.stability:
                        event = notify_event(name=name, confidence=conf, bbox=[int(x), int(y), int(w), int(h)], frame_bgr=frame)
                        logging.info("evento registrado: %s", event)
                        stable = 0
                        last_name = None
                else:
                    stable = 0
                    last_name = None

            if not args.no_gui:
                cv2.imshow("Golden Guard — Reconhecimento", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        logging.info("Interrompido pelo usuário.")
    finally:
        cap.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
