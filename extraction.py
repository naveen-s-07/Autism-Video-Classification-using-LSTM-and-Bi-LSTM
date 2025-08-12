# The following two methods are taken from this tutorial:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub

import main
import data_path
import data_path

def load_video(path, max_frames=0, resize=(main.IMG_SIZE, main.IMG_SIZE)):
    cap = main.cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = main.cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return main.np.array(frames)


def build_feature_extractor():
    feature_extractor = main.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(main.IMG_SIZE, main.IMG_SIZE, 3),
    )
    preprocess_input = main.keras.applications.inception_v3.preprocess_input

    inputs = main.keras.Input((main.IMG_SIZE, main.IMG_SIZE, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return main.keras.Model(inputs, outputs, name="feature_extractor")
feature_extractor=build_feature_extractor()


