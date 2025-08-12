import main
import extraction
import data_path
import label_display

def prepare_all_videos(df, root_dir):
    num_samples = len(df)
    video_paths = df["File"].values.tolist()  # Assuming "File" is the column with video names
    labels = df["Subfolder"].values.tolist()  # Assuming "Subfolder" is the column with labels

    # Create a StringLookup layer with num_oov_indices=1
    label_processor = main.keras.layers.StringLookup(
        num_oov_indices=1, vocabulary=main.np.unique(labels)
    )

    # Convert labels to indices using StringLookup
    labels = label_processor(labels)

    # `frame_masks` and `frame_features` are what we will feed to our sequence model.
    # `frame_masks` will contain a bunch of booleans denoting if a timestep is
    # masked with padding or not.
    frame_masks = main.np.zeros(shape=(num_samples, main.MAX_SEQ_LENGTH), dtype="bool")
    frame_features = main.np.zeros(
        shape=(num_samples, main.MAX_SEQ_LENGTH, main.NUM_FEATURES), dtype="float32"
    )

    # For each video.
    for idx, path in enumerate(video_paths):
        # Gather all its frames and add a batch dimension.
        frames = extraction.load_video(main.os.path.join(root_dir, path))
        frames = frames[None, ...]

        # Initialize placeholders to store the masks and features of the current video.
        temp_frame_mask = main.np.zeros(
            shape=(
                1,
                main.MAX_SEQ_LENGTH,
            ),
            dtype="bool",
        )
        temp_frame_features = main.np.zeros(
            shape=(1, main.MAX_SEQ_LENGTH, main.NUM_FEATURES), dtype="float32"
        )

        # Extract features from the frames of the current video.
        for i, batch in enumerate(frames):
            video_length = batch.shape[0]
            length = min(main.MAX_SEQ_LENGTH, video_length)
            for j in range(length):
                temp_frame_features[i, j, :] = extraction.feature_extractor.predict(
                    batch[None, j, :], verbose=0,
                )
            temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

        frame_features[idx,] = temp_frame_features.squeeze()
        frame_masks[idx,] = temp_frame_mask.squeeze()

    return (frame_features, frame_masks), labels


train_data, train_labels = prepare_all_videos(data_path.train_df, "train")
test_data, test_labels = prepare_all_videos(data_path.test_df, "test")

print(f"Frame features in train set: {train_data[0].shape}")
print(f"Frame masks in train set: {train_data[1].shape}")
