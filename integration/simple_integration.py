import os, json, time, pathlib

EVENTS_PATH = pathlib.Path(__file__).resolve().parent / "events.jsonl"
SNAP_DIR = pathlib.Path(__file__).resolve().parent / "snapshots"

def notify_event(name: str, confidence: float, bbox=None, frame_bgr=None):
    """
    Registra um evento de reconhecimento facial.
    - acrescenta uma linha JSON em integration/events.jsonl
    - salva um snapshot em integration/snapshots/
    - se WEBHOOK_URL estiver definido, envia o JSON via POST (best-effort)
    """
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")
    event = {
        "ts": ts,
        "name": name,
        "confidence": float(confidence),
        "bbox": bbox if bbox is not None else None,
        "action": "presence_logged"
    }

    # salva snapshot
    if frame_bgr is not None:
        try:
            import cv2
            snap_path = SNAP_DIR / f"{ts.replace(':','-')}_{name}.jpg"
            cv2.imwrite(str(snap_path), frame_bgr)
            event["snapshot"] = snap_path.name
        except Exception as e:
            event["snapshot_error"] = str(e)

    with open(EVENTS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")

    url = os.getenv("WEBHOOK_URL")
    if url:
        try:
            import requests
            requests.post(url, json=event, timeout=2)
        except Exception:
            pass

    return event
