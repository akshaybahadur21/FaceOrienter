import av
import cv2
import dlib
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
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


def setup_streamlit():
    st.set_page_config(page_title="Face Orienter", layout="centered", page_icon="😀 📐")
    image = Image.open('resources/face_orient.png')

    col1, col2, col3 = st.columns([3, 6, 3])
    with col1:
        st.write(' ')
    with col2:
        st.image(image, width=400, caption='Face Orientation using Computer Vision')
    with col3:
        st.write(' ')

    st.markdown(
        """
        <style>
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    apps = {
        "home": {"title": "Home", "icon": "house"},
        "face_orienter": {"title": "Face Orienter", "icon": "cloud-upload"},
        "about": {"title": "About", "icon": "activity"},
    }

    titles = [app["title"] for app in apps.values()]
    icons = [app["icon"] for app in apps.values()]
    params = st.experimental_get_query_params()

    if "page" in params:
        default_index = int(titles.index(params["page"][0].lower()))
    else:
        default_index = 0

    with st.sidebar:
        selected = option_menu(
            "Main Menu",
            options=titles,
            icons=icons,
            menu_icon="cast",
            default_index=default_index,
        )
    return selected


def orient():
    st.title("Application")
    st.markdown("""
            [![](https://img.shields.io/badge/GitHub-Source-brightgreen)](https://github.com/akshaybahadur21/FaceOrienter)
            """)

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


def home():
    st.title("Home")
    st.markdown("""
        [![](https://img.shields.io/badge/GitHub-Source-brightgreen)](https://github.com/akshaybahadur21/FaceOrienter)
        """)
    st.subheader("How to use Face Orienter")
    st.markdown("""
        - Click on `Face Orienter` tab
        - Select the device for webcam access
        - Click on the video icon and give permissions 
        """)


def main():
    selected = setup_streamlit()
    try:
        if selected.lower() == "home":
            home()
        elif selected.lower() == "face orienter":
            orient()
        elif selected.lower() == "about":
            pass
    except Exception as e:
        print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")


if __name__ == "__main__":
    main()
    col1, col2, col3 = st.columns([3, 6, 3])
    with col1:
        st.write(' ')
    with col2:
        st.markdown("""
                    ###### Made with ❤️ and 🦙 by [Akshay Bahadur](https://akshaybahadur.com)
                    """)
    with col3:
        st.write(' ')
