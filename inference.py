from main import np, MAX_SEQ_LENGTH, NUM_FEATURES, os, imageio
from extraction import feature_extractor, load_video
from label_display import label_processor
from Rnn_sequence_model import complex_rnn_model
from IPython.display import Image
from data_path import test_df
import pandas as pd
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(
        shape=(
            1,
            MAX_SEQ_LENGTH,
        ),
        dtype="bool",
    )
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")

    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = complex_rnn_model.predict([frame_features, frame_mask])[0]

    predicted_label = class_vocab[np.argmax(probabilities)]
    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames, predicted_label


# This utility is for visualization.
# Referenced from:
# https://www.tensorflow.org/hub/tutorials/action_recognition_with_tf_hub
def to_gif(images):
    converted_images = images.astype(np.uint8)
    imageio.mimsave("animation.gif", converted_images, duration=100)
    return Image("animation.gif")


def evaluate_all_videos(test_df):
    results = []

    for index, row in test_df.iterrows():
        video_path = row['File']
        true_label = row['Subfolder']
        frames, predicted_label = sequence_prediction(video_path)
        results.append((video_path, true_label, predicted_label))

    return results


def save_results_to_csv(results, file_path):
    results_df = pd.DataFrame(results, columns=['File', 'TrueLabel', 'PredictedLabel'])
    results_df.to_csv(file_path, index=False)


test_video = np.random.choice(test_df["File"].values.tolist())
print(f"Test video path: {test_video}")
test_frames, _ = sequence_prediction(test_video)
to_gif(test_frames[:MAX_SEQ_LENGTH])

results = evaluate_all_videos(test_df)
save_results_to_csv(results, "test_results.csv")

