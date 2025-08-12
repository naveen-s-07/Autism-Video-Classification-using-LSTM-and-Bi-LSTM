import main 
import extraction
import data_path
import pandas as pd

data_path.train_df = pd.read_csv('E:/Sreeraj/video_motion/RNN_complex_check/train.csv')
#print(data_path.train_df.columns)
#print(data_path.train_df.head())

data_path.train_df['tag'] = data_path.train_df['File'].apply(lambda x: x.split('_')[1])

# Print columns and head of the DataFrame (for debugging)
#print(data_path.train_df.columns)
# print(data_path.train_df.head())

# Build feature extractor (assuming this function is defined in extraction.py)
feature_extractor = extraction.build_feature_extractor()

# Create StringLookup for 'tag' column
label_processor = main.keras.layers.StringLookup(
    num_oov_indices=0,
    vocabulary=list(main.np.unique(data_path.train_df["tag"]))
)
# Print vocabulary mapped by StringLookup
print(label_processor.get_vocabulary())