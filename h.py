import cv2
import mediapipe as mp

# Load MediaPipe Hands Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Open Video Stream
cap = cv2.VideoCapture(0)  # 0 for webcam, you can provide video path for file

# Gesture labels
gesture_labels = {
    4: 'Thumbs Up', 
    8: 'Index Finger Pointing', 
    12: 'Middle Finger Pointing', 
    16: 'Ring Finger Pointing', 
    20: 'Little Finger Pointing'
}  # Landmark indices for different hand landmarks

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the image with MediaPipe Hands Model
    hands_results = hands.process(frame_rgb)

    # Do something with the results
    # For example, you can draw the landmarks or detect hand sign language gestures

    # Draw hand landmarks (optional)
    if hands_results.multi_hand_landmarks:
        for hand_landmarks in hands_results.multi_hand_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

            # Detect gestures
            for landmark_index, gesture in gesture_labels.items():
                landmark = hand_landmarks.landmark[landmark_index]
                landmark_x = landmark.x * frame.shape[1]
                landmark_y = landmark.y * frame.shape[0]

                if landmark_x < frame.shape[1] / 2:
                    cv2.putText(frame, gesture, (50, 50 + landmark_index * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Hand Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release Resources
cap.release()
cv2.destroyAllWindows()
