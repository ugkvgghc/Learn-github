from ultralytics import YOLO
import cv2
import winsound

# โหลดโมเดล
model = YOLO("best.pt")

# ===== เลือกกล้อง =====
# กล้องคอม
cap = cv2.VideoCapture(0)

# กล้องมือถือ (IP Webcam)
# cap = cv2.VideoCapture("http://10.12.59.53:8080/video")
# ======================

alerted = False  # กันเสียงดังรัว

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # วิเคราะห์ภาพ
    results = model(frame)

    found_no_helmet = False

    for box in results[0].boxes:
        cls = int(box.cls[0])
        label = model.names[cls].lower()  # แปลงเป็นตัวเล็ก

        # 🔴 ชื่อคลาส "ไม่ใส่หมวก"
        if label in ["without_helmet", "no_helmet"]:
            found_no_helmet = True

    # ถ้าเจอคนไม่ใส่หมวก → ส่งเสียง
    if found_no_helmet and not alerted:
        winsound.Beep(1000, 500)
        alerted = True

    # ถ้าไม่เจอแล้ว → รีเซ็ตเสียง
    if not found_no_helmet:
        alerted = False

    # วาดกรอบ
    annotated_frame = results[0].plot()
    cv2.imshow("Helmet Detection", annotated_frame)

    # ESC เพื่อออก
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()