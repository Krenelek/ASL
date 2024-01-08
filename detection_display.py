import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Camera setup
cap = cv2.VideoCapture(1)

# For storing recognized characters
recognized_characters = []

# mediapipe hands initialization
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    model_path = 'trained_model2d/model.h5'
    model = tf.keras.models.load_model(model_path)

    prev_time = time.time()

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Nie udalo sie zdobyc obrazu.")
            continue

        # Color conversion from BGR into RGB for camera feed
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hand and keypoints detection
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Keypoint vector
                hand_points = []
                for point in hand_landmarks.landmark:
                    hand_points.append(point.x)
                    hand_points.append(point.y)

                hand_points = np.array(hand_points)

                # Keypoint vector normalization
                hand_points = hand_points / np.max(hand_points)

                # Reshape keypoint vector
                hand_points = hand_points.astype(np.float32)
                hand_points = hand_points.reshape((1, 42))

                # Letter or digit recognition using trained model with a delay of 1 second
                current_time = time.time()
                if current_time - prev_time >= 1:
                    prediction = model.predict(hand_points)

                    # Get predicted label
                    if np.argmax(prediction) < 10:
                        prediction = chr(np.argmax(prediction) + 48)
                    else:
                        prediction = chr(np.argmax(prediction) + 65 - 10)

                    # Store recognized character
                    recognized_characters.append(prediction)

                    # Limit the history to the last 40 recognized characters
                    recognized_characters = recognized_characters[-40:]

                    prev_time = current_time

                # Render hand keypoints as lines in a new window along with the recognized letter/digit
                hand_image = np.zeros_like(image)
                mp_drawing.draw_landmarks(hand_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Displaying the hand rendering and recognition history
                history_image = np.zeros((hand_image.shape[0] + 300, hand_image.shape[1], 3), dtype=np.uint8)
                history_image[:hand_image.shape[0], :] = hand_image
                cv2.putText(history_image, ' '.join(recognized_characters), (10, hand_image.shape[0] + 250),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Displaying the camera feed window with added keypoints
                cv2.imshow('Camera Feed', cv2.resize(image, (400, 300)))

                # Displaying the hand recognition and history window
                cv2.imshow('Hand Recognition and History', cv2.resize(history_image, (600, 800)))

        # Close window by clicking on GUI close button [x]
        key = cv2.waitKey(1)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()
