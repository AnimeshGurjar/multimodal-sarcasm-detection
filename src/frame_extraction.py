import os
import cv2

# Paths
sarcasm_video_folder = "/s/babbage/b/nobackup/nblancha/public-datasets/ani/MUStARD/MUStARD/data/raw_videos/utterances_final"
context_video_folder = "/s/babbage/b/nobackup/nblancha/public-datasets/ani/MUStARD/MUStARD/data/raw_videos/context_final"
output_folder_sarcasm = "/s/babbage/b/nobackup/nblancha/public-datasets/ani/MUStARD/MUStARD/data/raw_videos/frames/sarcasm"
output_folder_context = "/s/babbage/b/nobackup/nblancha/public-datasets/ani/MUStARD/MUStARD/data/raw_videos/frames/context"

# Create output folders if they don't exist
os.makedirs(output_folder_sarcasm, exist_ok=True)
os.makedirs(output_folder_context, exist_ok=True)

def extract_frames(video_folder, output_folder, label):
    for video_file in os.listdir(video_folder):
        if video_file.endswith(".mp4"):
            video_path = os.path.join(video_folder, video_file)
            video_name = os.path.splitext(video_file)[0]

            # Load video
            cap = cv2.VideoCapture(video_path)
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = fps  # Extract 1 frame per second

            frame_count = 0
            extracted_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Extract frame every 'frame_interval'
                if frame_count % frame_interval == 0:
                    frame_filename = f"{video_name}_{label}_frame{extracted_count}.jpg"
                    frame_path = os.path.join(output_folder, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_count += 1

                frame_count += 1

            cap.release()

# Extract frames from both sarcasm & context videos
extract_frames(sarcasm_video_folder, output_folder_sarcasm, "sarcasm")
extract_frames(context_video_folder, output_folder_context, "context")

print("✅ Frame extraction complete for both sarcasm & context videos.")
