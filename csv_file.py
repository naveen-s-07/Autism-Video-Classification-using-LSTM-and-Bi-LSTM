import os
import csv

def create_csv(folder_path, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Sequence Number', 'File', 'Subfolder'])

        sequence_number = 1
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                subfolder = os.path.basename(root)
                csv_writer.writerow([sequence_number, file, subfolder])
                sequence_number += 1

# Path to the main data folder
data_folder = 'E:/Sreeraj/video_motion/RNN_complex_check'

# Paths for train and test folders
train_folder = os.path.join(data_folder, 'train')
test_folder = os.path.join(data_folder, 'test')

# Paths for train and test CSV files
train_csv_filename = 'E:/Sreeraj/video_motion/RNN_complex_check/train.csv'
test_csv_filename = 'E:/Sreeraj/video_motion/RNN_complex_check/test.csv'

# Creating CSV files for train and test data
create_csv(train_folder, train_csv_filename)
create_csv(test_folder, test_csv_filename)

print("CSV files created successfully.")