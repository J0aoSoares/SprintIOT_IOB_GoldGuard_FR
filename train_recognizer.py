import os, json, cv2, numpy as np

def load_faces(root="faces"):
    images, labels, names = [], [], []
    if not os.path.isdir(root):
        raise SystemExit("Pasta 'faces/' não encontrada.")
    people = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    if not people:
        raise SystemExit("Crie subpastas em 'faces/' (uma por pessoa) com imagens.")

    for idx, person in enumerate(people):
        pdir = os.path.join(root, person)
        names.append(person)
        for fn in os.listdir(pdir):
            if not fn.lower().endswith((".jpg",".jpeg",".png",".bmp")):
                continue
            path = os.path.join(pdir, fn)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
            if len(faces) == 0:
                # se não encontrou, usa a imagem inteira (último caso)
                face_roi = cv2.resize(img, (200,200))
                images.append(face_roi)
                labels.append(idx)
                continue
            # usa a MAIOR face detectada
            x, y, w, h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
            face_roi = cv2.resize(img[y:y+h, x:x+w], (200,200))
            images.append(face_roi)
            labels.append(idx)
    return images, np.array(labels), names

def main():
    images, labels, names = load_faces("faces")
    if len(images) < 2:
        raise SystemExit("Poucas imagens. Adicione mais fotos nas pastas de cada pessoa.")

    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
    recognizer.train(images, labels)

    os.makedirs("model", exist_ok=True)
    recognizer.write("model/lbph_model.xml")
    with open("model/labels.json", "w", encoding="utf-8") as f:
        json.dump({i: n for i, n in enumerate(names)}, f, ensure_ascii=False, indent=2)

    print("[ok] Treino concluído.")
    print(" - Modelo: model/lbph_model.xml")
    print(" - Labels:", names)

if __name__ == "__main__":
    main()
