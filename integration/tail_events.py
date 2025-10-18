import time, pathlib
EVENTS = pathlib.Path(__file__).resolve().parent / "events.jsonl"
print(f"Tailing {EVENTS}... (Ctrl+C to stop)")
EVENTS.touch(exist_ok=True)
with open(EVENTS, "r", encoding="utf-8") as f:
    f.seek(0, 2)
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.25)
            continue
        print(line.strip())
