# import cv2
# import mediapipe as mp
# import numpy as np
# import tensorflow as tf
# import time
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# # Initialize Mediapipe Hands
# mp_drawing = mp.solutions.drawing_utils
# mp_hands = mp.solutions.hands

# # Camera setup (select device)
# cap = cv2.VideoCapture(0)

# # For storing recognized characters
# recognized_characters = []

# # Set up a global variable for the figure and initial plots
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter([], [], [], c='r', marker='o')
# lines = [ax.plot([], [], [], c='b')[0] for _ in range(len(mp_hands.HAND_CONNECTIONS))]

# # mediapipe hands initialization
# with mp_hands.Hands(
#         static_image_mode=False,
#         max_num_hands=1,
#         min_detection_confidence=0.5,
#         min_tracking_confidence=0.5) as hands:

#     model_path = 'trained_model2d/model.h5'
#     model = tf.keras.models.load_model(model_path)

#     prev_time = time.time()

#     while cap.isOpened():
#         success, image = cap.read()
#         if not success:
#             print("Unable to capture image.")
#             continue

#         # Color conversion from BGR into RGB for camera feed
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         # Hand and keypoints detection
#         results = hands.process(image_rgb)
#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # 3D hand skeleton visualization
#                 hand_points = np.array([[point.x, point.y, -point.z] for point in hand_landmarks.landmark])

#                 # Normalize hand points
#                 min_values = np.min(hand_points, axis=0)
#                 max_values = np.max(hand_points, axis=0)
#                 scaling_factors = 2 / (max_values - min_values)
#                 normalized_hand_points = (hand_points - min_values) * scaling_factors - 1

#                 # Update scatter plot for points
#                 sc._offsets3d = (normalized_hand_points[:, 0], normalized_hand_points[:, 1], normalized_hand_points[:, 2])

#                 # Update line plot for connecting lines
#                 for i, connection in enumerate(mp_hands.HAND_CONNECTIONS):
#                     connection_points = normalized_hand_points[np.array(connection)].T
#                     lines[i].set_data(connection_points[:2, :])
#                     lines[i].set_3d_properties(connection_points[2, :])

#                 # Set appropriate axis limits for better visualization
#                 ax.set_xlim([-1, 1])
#                 ax.set_ylim([-1, 1])
#                 ax.set_zlim([-1, 1])

#                 # Display the 3D skeleton window
#                 plt.draw()
#                 plt.pause(0.01)

#         # Display the camera feed window with added keypoints
#         cv2.imshow('Camera Feed', cv2.resize(image, (400, 300)))

#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the webcam and close windows
#     cap.release()
#     cv2.destroyAllWindows()




import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize Mediapipe Hands
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Camera setup (select device)
cap = cv2.VideoCapture(0)

# For storing recognized characters
recognized_characters = []

# Set up a global variable for the figure and initial plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter([], [], [], c='r', marker='o')
lines = [ax.plot([], [], [], c='b')[0] for _ in range(len(mp_hands.HAND_CONNECTIONS))]

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
            print("Unable to capture image.")
            continue

        # Color conversion from BGR into RGB for camera feed
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Hand and keypoints detection
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 3D hand skeleton visualization
                hand_points_3d = np.array([[point.x, point.y, -point.z] for point in hand_landmarks.landmark])

                # Normalize hand points
                min_values = np.min(hand_points_3d, axis=0)
                max_values = np.max(hand_points_3d, axis=0)
                scaling_factors = 2 / (max_values - min_values)
                normalized_hand_points = (hand_points_3d - min_values) * scaling_factors - 1

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

                # Update scatter plot for points
                sc._offsets3d = (normalized_hand_points[:, 0], normalized_hand_points[:, 1], normalized_hand_points[:, 2])

                # Update line plot for connecting lines
                for i, connection in enumerate(mp_hands.HAND_CONNECTIONS):
                    connection_points = normalized_hand_points[np.array(connection)].T
                    lines[i].set_data(connection_points[:2, :])
                    lines[i].set_3d_properties(connection_points[2, :])

                # Set appropriate axis limits for better visualization
                ax.set_xlim([-1, 1])
                ax.set_ylim([-1, 1])
                ax.set_zlim([-1, 1])

                # Set the title with the recognized characters
                ax.set_title(' '.join(recognized_characters[-10:]))

                # Display the 3D skeleton window
                plt.draw()
                plt.pause(0.01)


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

                    # Limit the history to the last 10 recognized characters
                    recognized_characters = recognized_characters[-10:]

                    prev_time = current_time

        # Display the camera feed window with added keypoints
        cv2.imshow('Camera Feed', cv2.resize(image, (400, 300)))

        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q') or not cv2.getWindowProperty('Camera Feed', cv2.WND_PROP_VISIBLE):
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
