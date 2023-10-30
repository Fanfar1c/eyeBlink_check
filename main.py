from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time

eyes_closed_time = 0  # Время, когда глаза закрыты
eyes_open = True  # Флаг, указывающий, что глаза открыты
text_duration = 5
show_text = False
show_text_start_time = 0

base_options = python.BaseOptions(model_asset_path='model/face_landmarker_v2_with_blendshapes.task')
options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)


def draw_landmarks_on_image(rgb_image, detection_result):
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image



cap = cv2.VideoCapture(0)
while True:
    succes, img = cap.read()
    
    if not succes:
        print("Не удалось захватить кадр с веб-камеры.")
        continue

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=imgRGB)

    detection_result = detector.detect(mp_image)
    eyeBlinkLeft = detection_result.face_blendshapes[0][9]
    eyeBlinkRight = detection_result.face_blendshapes[0][10]

    if eyeBlinkLeft.score >= 0.50 or eyeBlinkRight.score >= 0.50:
        if eyes_open:
            eyes_closed_start_time = time.time()
            eyes_open = False
    else:
        if not eyes_open:
            eyes_closed_time = time.time() - eyes_closed_start_time
            if eyes_closed_time >= 2:
                message = "Глаза закрыты более 2 секунд. Вероятно, человек спит."
                print(message)
                show_text = True
                show_text_start_time = time.time()
            eyes_open = True
    
    if show_text:
        message = "Глаза закрыты более 2 секунд. Вероятно, человек спит."
        cv2.putText(annotated_image, message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if time.time() - show_text_start_time >= text_duration:
            show_text = False

    annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
   

    cv2.imshow('Face Detection', cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    # Прерывание цикла при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Завершение захвата и закрытие окон
cap.release()
cv2.destroyAllWindows()