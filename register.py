from pathlib import Path

import cv2

DATABASE = Path("database")

def register_user():
    name = input("Enter user's name: ").strip()

    if not name:
        print("Name cannot be empty.")
        return

    if (DATABASE / name).exists():
        print("User is already registered.")
        return

    user_dir = DATABASE / name
    user_dir.mkdir(parents=True, exist_ok=True)

    camera = cv2.VideoCapture(0)
    count = 0

    print("Press SPACE to capture a picture.")
    print("Press Q to finish registration.")

    while True:
        success, frame = camera.read()

        if not success:
            break

        cv2.putText(
            frame,
            f"Captured: {count}/5",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("User Registration", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord(" ") and count < 5:
            filename = user_dir / f"{count + 1}.jpg"
            cv2.imwrite(str(filename), frame)
            count += 1
            print(f"Saved {filename}")

    camera.release()
    cv2.destroyAllWindows()

    if count == 0:
        user_dir.rmdir()
        print("Registration cancelled.")
    else:
        print(f"Registration completed: {name}")


if __name__ == "__main__":
    register_user()

