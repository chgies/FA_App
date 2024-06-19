import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

def run_loop():
    global options
    with vision.HandLandmarker.create_from_options(options) as landmarker:        
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            success, image = cap.read()
            #start_time = time.time()
            if not success:
                break
            
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            if len(result.hand_landmarks) == 0:
                print("No hand landmarkers found in this frame")
            output_window = draw_landmarks_on_image(image, result)
            if output_window is not None:
                cv2.imshow("MediaPipe Hand Landmark", output_window)
            else:            
                cv2.imshow("MediaPipe Hand Landmark", image)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return -1
            #end_time = time.time()
            #print(f"FPS of video: {1.0 / (end_time-start_time)}")
            #frame += 1

def create_landmark_options():
    """
    Define the options needed by Mediapipe Pose Landmarker.
        Params: None
        Returns: options (mediapipe.tasks.python.vision.PoseLandmarkerOptions): The defined options
    """
    landmark_path = "./mediapipe_landmarker/hand_landmarker.task"
    base_options = python.BaseOptions(model_asset_path=landmark_path, delegate="GPU")
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    return options

options = create_landmark_options()
run_loop()
