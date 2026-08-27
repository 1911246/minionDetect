import cv2
import mediapipe as mp
import numpy as np

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

happy_img = cv2.imread("images/happyMin.png", cv2.IMREAD_UNCHANGED)
surprised_img = cv2.imread("images/surpiseMin.png", cv2.IMREAD_UNCHANGED)
angry_img = cv2.imread("images/angryMin.png", cv2.IMREAD_UNCHANGED)

# Path to MediaPipe model
model_path = "face_landmarker.task"


# Create Face Landmarker
base_options = python.BaseOptions(
    model_asset_path=model_path
)

options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    num_faces=1
)

detector = vision.FaceLandmarker.create_from_options(options)


# Open webcam
camera = cv2.VideoCapture(0)

#distance calc
def distance(p1, p2):
    return np.sqrt(
        (p1.x - p2.x) ** 2 +
        (p1.y - p2.y) ** 2
    )

#happy 
def happy(landmarks):

    # Mouth
    left_corner = landmarks[61]
    right_corner = landmarks[291]

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    mouth_width = distance(left_corner, right_corner)
    mouth_height = distance(upper_lip, lower_lip)

    smiling = mouth_width > mouth_height * 15

    # Eyebrows
    left_eyebrow = landmarks[105]
    right_eyebrow = landmarks[334]

    # Eyes
    left_eye = landmarks[23]
    right_eye = landmarks[253]

    left_brow_eye = distance(left_eyebrow, left_eye)
    right_brow_eye = distance(right_eyebrow, right_eye)

    eyebrows_raised = (
        left_brow_eye > 0.085 and
        right_brow_eye > 0.085
    )

    return eyebrows_raised

#surpisedd
def surprised(landmarks):

    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    mouth_height = distance(upper_lip, lower_lip)

    return mouth_height > 0.04

#angry
def angry(landmarks):

    # Eyebrows
        left_eyebrow = landmarks[105]
        right_eyebrow = landmarks[334]
    
        # Eyes
        left_eye = landmarks[23]
        right_eye = landmarks[253]
    
        left_brow_eye = distance(left_eyebrow, left_eye)
        right_brow_eye = distance(right_eyebrow, right_eye)
    
        eyebrows_down = (
            left_brow_eye < 0.05 and
            right_brow_eye < 0.05
        )
    
        return eyebrows_down
    
def get_expression(landmarks):

    if surprised(landmarks):
        return "surprised"

    elif happy(landmarks):
        return "Happy"

    elif angry(landmarks):
        return "ANGRY"

    else:
        return "NEUTRAL"




while True:

    ret, frame = camera.read()

    if not ret:
        print("Could not access camera")
        break

    # OpenCV gives us BGR
    # MediaPipe wants RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Detect face landmarks
    results = detector.detect(mp_image)

    # Draw landmarks
    if results.face_landmarks:

        for face_landmarks in results.face_landmarks:

            # Determine expression
            expression = get_expression(face_landmarks)

            if expression=="surprised" :
                cv2.imshow("Minion",surprised_img)

        
            if expression=="Happy" :
                cv2.imshow("Minion",happy_img)

            if expression=="ANGRY" :
                cv2.imshow("Minion",angry_img)

            

            # Display expression
            cv2.putText(
                frame,
                expression,
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            for i, landmark in enumerate(face_landmarks):

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    1,
                    (0, 255, 0),
                    -1
                )

                # Show landmark number
                cv2.putText(
                                  frame,
                                  str(i),
                                  (x, y),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  0.3,
                                  (255, 255, 255),
                                  1
                              )



                
                

        

    cv2.imshow("Minion Detector", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    


camera.release()
cv2.destroyAllWindows()