import av
import cv2
import dlib
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    RTCConfiguration,
    WebRtcMode,
    webrtc_streamer,
)

detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor(
    "model/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def main():
    st.header("WebRTC demo")

    pages = {
        "Consuming media files on server-side and streaming it to browser (recvonly)": app_object_detection,
        # noqa: E501
    }
    page_titles = pages.keys()

    page_title = st.sidebar.selectbox(
        "Choose the app mode",
        page_titles,
    )
    st.subheader(page_title)

    page_func = pages[page_title]
    page_func()

    st.sidebar.markdown(
        """
---
<a href="https://www.buymeacoffee.com/whitphx" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" width="180" height="50" ></a>
    """,  # noqa: E501
        unsafe_allow_html=True,
    )


def app_object_detection():
    def draw_line(frame, a, b, color=(255, 255, 0)):
        cv2.line(frame, a, b, color, 10)

    def _annotate_image(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        subjects = detect(gray, 0)
        for subject in subjects:
            landmarks = predict(gray, subject)

            size = frame.shape
            # 2D image points. If you change the image, you need to change vector
            image_points = np.array([
                (landmarks.part(33).x, landmarks.part(33).y),  # Nose tip
                (landmarks.part(8).x, landmarks.part(8).y),  # Chin
                (landmarks.part(36).x, landmarks.part(36).y),  # Left eye left corner
                (landmarks.part(45).x, landmarks.part(45).y),  # Right eye right corne
                (landmarks.part(48).x, landmarks.part(48).y),  # Left Mouth corner
                (landmarks.part(54).x, landmarks.part(54).y)  # Right mouth corner
            ], dtype="double")

            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-225.0, 170.0, -135.0),  # Left eye left corner
                (225.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner

            ])

            focal_length = size[1]
            center = (size[1] / 2, size[0] / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]],
                 [0, focal_length, center[1]],
                 [0, 0, 1]], dtype="double"
            )

            dist_coeffs = np.zeros((4, 1))
            (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                          dist_coeffs)

            (b1, jacobian) = cv2.projectPoints(np.array([(350.0, 270.0, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (b2, jacobian) = cv2.projectPoints(np.array([(-350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)
            (b3, jacobian) = cv2.projectPoints(np.array([(-350.0, 270, 0.0)]), rotation_vector, translation_vector,
                                               camera_matrix, dist_coeffs)
            (b4, jacobian) = cv2.projectPoints(np.array([(350.0, -270.0, 0.0)]), rotation_vector,
                                               translation_vector, camera_matrix, dist_coeffs)

            (b11, jacobian) = cv2.projectPoints(np.array([(450.0, 350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b12, jacobian) = cv2.projectPoints(np.array([(-450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b13, jacobian) = cv2.projectPoints(np.array([(-450.0, 350, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)
            (b14, jacobian) = cv2.projectPoints(np.array([(450.0, -350.0, 400.0)]), rotation_vector,
                                                translation_vector, camera_matrix, dist_coeffs)

            b1 = (int(b1[0][0][0]), int(b1[0][0][1]))
            b2 = (int(b2[0][0][0]), int(b2[0][0][1]))
            b3 = (int(b3[0][0][0]), int(b3[0][0][1]))
            b4 = (int(b4[0][0][0]), int(b4[0][0][1]))

            b11 = (int(b11[0][0][0]), int(b11[0][0][1]))
            b12 = (int(b12[0][0][0]), int(b12[0][0][1]))
            b13 = (int(b13[0][0][0]), int(b13[0][0][1]))
            b14 = (int(b14[0][0][0]), int(b14[0][0][1]))

            draw_line(frame, b1, b3)
            draw_line(frame, b3, b2)
            draw_line(frame, b2, b4)
            draw_line(frame, b4, b1)

            draw_line(frame, b11, b13)
            draw_line(frame, b13, b12)
            draw_line(frame, b12, b14)
            draw_line(frame, b14, b11)

            draw_line(frame, b11, b1, color=(0, 255, 0))
            draw_line(frame, b13, b3, color=(0, 255, 0))
            draw_line(frame, b12, b2, color=(0, 255, 0))
            draw_line(frame, b14, b4, color=(0, 255, 0))
        return frame

    def callback(frame: av.VideoFrame) -> av.VideoFrame:
        frame = frame.to_ndarray(format="bgr24")
        annotated_image = _annotate_image(frame)
        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_frame_callback=callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.markdown(
        "This demo uses a model and code from "
        "https://github.com/robmarkcole/object-detection-app. "
        "Many thanks to the project."
    )


if __name__ == "__main__":
    main()
