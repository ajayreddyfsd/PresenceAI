import cv2
import mediapipe as mp
import math
import time
from pymongo import MongoClient
from datetime import datetime


def calculate_angle(a, b, c):
    ab = (a[0] - b[0], a[1] - b[1])
    cb = (c[0] - b[0], c[1] - b[1])

    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = math.sqrt(cb[0] ** 2 + cb[1] ** 2)

    if mag_ab * mag_cb == 0:
        return 0

    angle_rad = math.acos(dot / (mag_ab * mag_cb))
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def store_to_mongo(metrics):
    mongo_uri = "mongodb://localhost:27017"
    client = MongoClient(mongo_uri)

    db = client["presence_ai"]
    collection = db["body_metrics"]

    result = collection.insert_one(metrics)
    print(f"\n\u2714\ufe0f Metrics stored to MongoDB with ID: {result.inserted_id}")


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture('./good-mini.mp4')

    posture_scores = []
    gesture_scores = []
    expressive_gestures = 0
    eye_contact_frames = 0
    total_frames = 0
    start_time = time.time()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
            mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
            mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            total_frames += 1
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            pose_results = pose.process(image)
            hand_results = hands.process(image)
            face_results = face.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark

                left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
                left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

                shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
                hip_mid = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)
                point_above_shoulders = (shoulder_mid[0], shoulder_mid[1] - 0.1)

                posture_angle = calculate_angle(hip_mid, shoulder_mid, point_above_shoulders)
                if posture_angle > 160:
                    posture_scores.append(100)
                elif posture_angle > 140:
                    posture_scores.append(70)
                else:
                    posture_scores.append(30)

                hand_raised = False
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST.value]
                        if wrist.y < shoulder_mid[1]:
                            hand_raised = True
                            break
                if hand_raised:
                    expressive_gestures += 1
                    gesture_scores.append(100)
                else:
                    gesture_scores.append(30)

            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0].landmark
                left_iris = face_landmarks[468]  # left iris
                nose_tip = face_landmarks[1]
                if abs(left_iris.x - nose_tip.x) < 0.02:
                    eye_contact_frames += 1

            cv2.imshow('PresenceAI - Live Feedback', image)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    end_time = time.time()

    avg_posture_score = sum(posture_scores) / len(posture_scores) if posture_scores else 0
    avg_gesture_score = sum(gesture_scores) / len(gesture_scores) if gesture_scores else 0
    final_score = (avg_posture_score + avg_gesture_score) / 2
    eye_contact_percent = (eye_contact_frames / total_frames) * 100 if total_frames > 0 else 0
    total_duration = end_time - start_time

    print(f"Average Posture Score: {avg_posture_score:.1f}")
    print(f"Average Gesture Score: {avg_gesture_score:.1f}")
    print(f"Final Body Language Score: {final_score:.1f}")
    print(f"Eye Contact %: {eye_contact_percent:.1f}%")
    print(f"Expressive Gestures: {expressive_gestures}")
    print(f"Total Duration: {total_duration:.1f} seconds")

    metrics = {
        "timestamp": datetime.utcnow(),
        "avg_posture_score": round(avg_posture_score, 1),
        "avg_gesture_score": round(avg_gesture_score, 1),
        "final_composite_score": round(final_score, 1),
        "expressive_gestures": expressive_gestures,
        "eye_contact_percent": round(eye_contact_percent, 1),
        "duration_seconds": round(total_duration, 1)
    }
    store_to_mongo(metrics)


if __name__ == "__main__":
    main()