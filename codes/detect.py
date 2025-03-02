import os
import cv2
from ultralytics import YOLO, SAM

# Load Models
yolo_model = YOLO("models/best_yolo10_40k.pt")
sam_model = SAM("models/mobile_sam.pt")

# Define Input/Output Folders
input_folder = "input_videos/"
output_folder = "output/"
os.makedirs(output_folder, exist_ok=True)

# Process Videos
for video_name in os.listdir(input_folder):
    video_path = os.path.join(input_folder, video_name)
    print(f"Processing: {video_name}")
    
    # Run YOLO Detection
    results = yolo_model.predict(video_path, conf=0.7)
    
    # Run SAM Segmentation
    for result in results:
        mask = sam_model.predict(result.orig_img, result.boxes.xyxy.tolist())
        
        # Save Mask
        mask_path = os.path.join(output_folder, f"mask_{video_name}.png")
        cv2.imwrite(mask_path, mask)

print("✅ Processing Complete!")