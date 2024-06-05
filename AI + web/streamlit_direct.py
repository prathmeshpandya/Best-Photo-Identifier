# importing required libraries
import streamlit as st
import cv2
import mediapipe as mp
from deepface import DeepFace
import numpy as np

num = st.number_input(
    "Enter number of images to be generated",
    min_value=1,
    max_value=100,
    value=10,
    step=1,
)


uploaded_file = st.file_uploader("Choose a video file", type=["mp4"])
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()

    with open("saved_video.mp4", "wb") as f:
        f.write(bytes_data)

    cap = cv2.VideoCapture("saved_video.mp4")
    score_arr = []

    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5
    )

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to grayscale for blur detection and contrast calculation
        gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        score = 0

        # Face detection
        results1 = face_detection.process(image_rgb)
        if results1.detections:
            score += 2
        else:
            score -= 1

        # Pose detection
        results2 = pose.process(image_rgb)
        if results2.pose_landmarks:
            score += 1
        else:
            score -= 1

        # Blur detection
        laplacian_var = cv2.Laplacian(gray_image, cv2.CV_64F).var()
        threshold = 100
        if laplacian_var < threshold:
            score += 1
        else:
            score -= 1

        # Contrast score
        mean_intensity = np.mean(gray_image)
        std_intensity = np.std(gray_image)
        try:
            contrast_score = int((std_intensity / mean_intensity) * 100)
            if 40 < contrast_score < 60:
                score += 1
            else:
                score -= 1
        except:
            pass

        # Expression detection
        result4 = DeepFace.analyze(
            image_rgb, actions=["emotion"], enforce_detection=False
        )

        possible_emotion = ["angry", "happy", "sad", "neutral"]
        for attri in result4:
            emotion = attri["dominant_emotion"]
            if emotion in possible_emotion:
                if emotion == "happy":
                    score += 2
                elif emotion == "sad" or emotion == "angry":
                    score -= 1
                elif emotion == "neutral":
                    score += 1

        score_arr.append((score, frame))

    # Sort the score_arr based on scores in descending order
    sorted_arr = sorted(score_arr, key=lambda x: x[0], reverse=True)

    # Save the best 15 frames
    for i in range(int(num)):
        score, best_frame = sorted_arr[i]
        cv2.imwrite(
            f"C:\\Users\Kushal\\Downloads\\dataset\\results\\final_image{i+1}.jpg",
            best_frame,
        )
        best_frame = cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR)
        st.image(best_frame)
    # Release the video capture
    cap.release()
