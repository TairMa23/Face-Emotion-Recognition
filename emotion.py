import cv2
from fer import FER

cap = cv2.VideoCapture(0)
detector = FER()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Emotion detection
    result = detector.detect_emotions(frame)

    if result:
        # Draw rectangle and emotion for each detected face
        for face in result:
            bounding_box = face['box']
            emotions = face['emotions']

            x, y, w, h = bounding_box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            emotion = max(emotions, key=emotions.get)
            score = emotions[emotion]

            cv2.putText(frame, f"{emotion}: {score:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
