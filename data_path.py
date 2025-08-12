import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator
train_df = pd.read_csv("E:/Sreeraj/video_motion/RNN_complex_check/train.csv")
test_df = pd.read_csv("E:/Sreeraj/video_motion/RNN_complex_check/test.csv")

print(f"Total videos for training: {len(train_df)}")
print(f"Total videos for testing: {len(test_df)}")

print(train_df.sample(10))

# Check the distribution of classes
train_df['File'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.show()



datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)