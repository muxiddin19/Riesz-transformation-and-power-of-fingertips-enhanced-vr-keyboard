import cv2

for i in range(6):
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    opened = cap.isOpened()
    print(f"index {i}: opened={opened}")

    if opened:
        ok, frame = cap.read()
        if ok:
            print(f"          got frame, shape={frame.shape}")
        else:
            print("          opened but no frame")

    cap.release()