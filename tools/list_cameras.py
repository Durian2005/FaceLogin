import cv2

print("opencv", cv2.__version__)
for idx in range(5):
    cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
    opened = cap.isOpened()
    info = ""
    if opened:
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        backend = cap.getBackendName()
        info = f" {int(w)}x{int(h)} backend={backend}"
        ok, _ = cap.read()
        info += f" read={ok}"
    cap.release()
    print(f"index {idx}: opened={opened}{info}")
