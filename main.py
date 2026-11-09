import queue
import threading
import time
from pathlib import Path

import cv2
from deepface import DeepFace

DATABASE = Path("database")

face_detector = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)

face_queue = queue.Queue(maxsize=1)

result_lock = threading.Lock()

latest_result = {
    "name": "Analyzing...",
    "emotion": "Analyzing...",
    "updated": 0
}

running = True


def analyze_face(face):
    """Run AI recognition and emotion detection."""

    name = "Unknown Person"
    emotion = "Unknown"

    try:
        # Face recognition
        matches = DeepFace.find(
            img_path=face,
            db_path=str(DATABASE),
            model_name="ArcFace",
            detector_backend="opencv",
            distance_metric="cosine",
            threshold=0.30,
            enforce_detection=False,
            silent=True
        )

        if matches and not matches[0].empty:
            best_match = matches[0].iloc[0]

            identity = best_match["identity"]
            distance = best_match["distance"]

            name = Path(identity).parent.name

            print(
                f"Recognized: {name} | "
                f"Distance: {distance:.4f}"
            )

    except Exception as error:
        print("Recognition error:", error)

    try:
        # Emotion detection
        analysis = DeepFace.analyze(
            img_path=face,
            actions=["emotion"],
            detector_backend="skip",
            enforce_detection=False,
            silent=True
        )

        if analysis:
            emotion = analysis[0]["dominant_emotion"].capitalize()

    except Exception as error:
        print("Emotion error:", error)

    with result_lock:
        latest_result.update({
            "name": name,
            "emotion": emotion,
            "updated": time.time()
        })


def ai_worker():
    """Process AI requests without blocking webcam."""

    while running:
        try:
            face = face_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        try:
            analyze_face(face)
        finally:
            face_queue.task_done()


def main():
    global running

    if not DATABASE.exists():
        print("No registered users found.")
        return

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    # Start AI processing thread
    worker = threading.Thread(
        target=ai_worker,
        daemon=True
    )
    worker.start()

    last_submit = 0
    print("Real-time recognition started. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                break

            # Mirror webcam
            frame = cv2.flip(frame, 1)

            gray = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2GRAY
            )

            # Detect faces on every frame
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.2,
                minNeighbors=5,
                minSize=(80, 80)
            )

            for (x, y, w, h) in faces:

                margin = 20

                x1 = max(0, x - margin)
                y1 = max(0, y - margin)
                x2 = min(frame.shape[1], x + w + margin)
                y2 = min(frame.shape[0], y + h + margin)

                face_crop = frame[y1:y2, x1:x2].copy()

                now = time.time()

                if (
                    now - last_submit >= 0.8
                    and face_queue.empty()
                ):
                    try:
                        face_queue.put_nowait(face_crop)
                        last_submit = now
                    except queue.Full:
                        pass

                with result_lock:
                    result = latest_result.copy()

                # Avoid displaying stale predictions
                if now - result["updated"] > 4:
                    name = "Analyzing..."
                    emotion = "Analyzing..."
                else:
                    name = result["name"]
                    emotion = result["emotion"]

                color = (
                    (0, 255, 0)
                    if name != "Unknown Person"
                    else (0, 165, 255)
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    name,
                    (x1, max(25, y1 - 35)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Emotion: {emotion}",
                    (x1, y2 + 25 if y2 + 25 < frame.shape[0] else y2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.imshow("AI Face Recognition System", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        running = False
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
