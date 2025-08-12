import cv2

def get_video_frame_size(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return None

    # Get the width and height of the frames
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Release the video capture object
    cap.release()

    return frame_width, frame_height

# Example usage
video_path = "E:/Sreeraj/video_motion/test/v_CricketShot_g01_c01.avi"
frame_size = get_video_frame_size(video_path)

if frame_size:
    print(f"Frame width: {frame_size[0]}")
    print(f"Frame height: {frame_size[1]}")