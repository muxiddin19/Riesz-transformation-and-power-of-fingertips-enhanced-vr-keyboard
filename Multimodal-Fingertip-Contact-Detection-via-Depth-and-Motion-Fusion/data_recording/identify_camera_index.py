
import cv2

CANDIDATE_INDICES = [0, 3]  # from your find_droidcam_index.py results

for i in CANDIDATE_INDICES:
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print(f"index {i}: could not open")
        continue

    ok, frame = cap.read()
    cap.release()

    if not ok:
        print(f"index {i}: opened but no frame")
        continue

    filename = f"snapshot_index_{i}.png"
    cv2.imwrite(filename, frame)
    print(f"index {i}: saved {filename} ({frame.shape[1]}x{frame.shape[0]})")

print("\nOpen the saved PNGs and check which one shows your keyboard sheet (phone).")