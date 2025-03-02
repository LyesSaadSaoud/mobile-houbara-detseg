import os
import cv2
import glob
import time
import numpy as np
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from ultralytics import YOLO, SAM

# ========== 🚀 Load Models ==========
# Ensure models exist in 'models' directory before running the script
yolo_model = YOLO("models/best_yolo10_40k.pt")
sam_model = SAM("models/mobile_sam.pt")

# ========== 📂 Define Directories ==========
input_folder = "input_videos/"
output_folder = "output/"
output_folder_box = os.path.join(output_folder, "yolo_boxes")
output_folder_seg = os.path.join(output_folder, "sam_masks")
predict_folder = os.path.join(output_folder, "yolo_predictions")

# ========== 🏗️ Ensure Folders Exist ==========
folders = [output_folder, output_folder_box, output_folder_seg, predict_folder]
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Number of SAM worker threads
NUM_SAM_WORKERS = 4


# ========== 🚀 YOLO Inference Function ==========
def run_yolo(input_queue):
    """ Runs YOLO on images and puts results into a queue for SAM processing. """
    print("🔄 Running YOLO...")
    start_time = time.time()

    yolo_results = yolo_model.predict(
        input_folder,
        save=True,
        conf=0.7,
        project=predict_folder,
        name="exp",
        exist_ok=True
    )

    total_yolo_time = time.time() - start_time  # Total YOLO inference time

    # Locate YOLO output folder
    yolo_output_folder = os.path.join(predict_folder, "exp")
    if not os.path.exists(yolo_output_folder):
        raise FileNotFoundError(f"❌ YOLO output folder not found: {yolo_output_folder}")

    predicted_images = glob.glob(os.path.join(yolo_output_folder, "*.jpg"))
    for result in yolo_results:
        filename = os.path.basename(result.path)
        bboxes = result.boxes.xyxy.tolist()

        if not bboxes:  # Check if YOLO found anything
            print(f"⚠️ No objects detected in {filename}. Skipping SAM.")
            continue

        print(f"✅ YOLO detected {len(bboxes)} objects in {filename}")

    for result in yolo_results:
        filename = os.path.basename(result.path)
        yolo_result_img_path = next((img for img in predicted_images if filename in img), None)

        if not yolo_result_img_path:
            print(f"⚠️ Warning: No YOLO output found for {filename}")
            continue

        print(f"✅ YOLO processed: {filename} ➡️ Sending to SAM Queue")
        input_queue.put((filename, result.boxes.xyxy.tolist(), yolo_result_img_path))

    print(f"✅ Total YOLO Time: {total_yolo_time:.2f} sec")

    # 🔥 Ensure every SAM worker receives `None` to stop processing
    for _ in range(NUM_SAM_WORKERS):
        input_queue.put(None)


# ========== 🎯 SAM Inference Function ==========
def run_sam(input_queue):
    """ Runs SAM on bounding boxes received from YOLO and saves masks. """
    import threading
    thread_name = threading.current_thread().name
    print(f"🟣 SAM Worker {thread_name} Started")

    while True:
        try:
            item = input_queue.get(timeout=10)  # Prevents infinite blocking
        except:
            print(f"❌ SAM Worker {thread_name} Queue Timeout. Waiting for more tasks...")
            continue

        if item is None:
            print(f"🛑 SAM Worker {thread_name} received stop signal. Exiting...")
            break

        filename, bboxes, yolo_img_path = item
        print(f"🔵 SAM Processing {filename} with {len(bboxes)} boxes in {thread_name}")

        if not bboxes:
            print(f"⚠️ No bounding boxes detected for {filename}. Skipping SAM.")
            continue

        # Load image for segmentation
        image = cv2.imread(yolo_img_path)
        if image is None:
            print(f"❌ Error: Could not read {yolo_img_path}. Skipping.")
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert for PIL and SAM
        start_time = time.time()

        # Process each bounding box
        for idx, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = map(int, bbox)

            # Check if bounding box values are valid
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                print(f"⚠️ Invalid bbox {bbox} in {filename}. Skipping.")
                continue

            print(f"🔍 Running SAM on {filename}, bbox {idx}: {bbox}")

            # Format bounding box for SAM (adding dummy class label & score)
            sam_bbox = [[x1, y1, x2, y2, 0, 1.0]]  # [x1, y1, x2, y2, class_id, confidence]

            # Run SAM and ensure correct format
            try:
                mask_gen = sam_model.predict(image_rgb, sam_bbox)  # Run SAM
                masks = list(mask_gen)  # Convert generator to list

                if not masks or masks[0] is None:
                    print(f"⚠️ No mask generated for {filename}, bbox {idx}.")
                    continue

                # Extract the most confident mask (SAM may return multiple masks)
                mask = masks[0]  # Taking the first mask (Modify this logic if needed)

                # Ensure mask is a valid 2D NumPy array
                if isinstance(mask, list):  # If SAM returns a list of masks
                    mask = np.array(mask[0], dtype=np.uint8)  # Convert first mask
                else:
                    mask = np.array(mask, dtype=np.uint8)

                # Ensure it's a binary mask
                mask = (mask > 0).astype(np.uint8) * 255

                # Save mask image
                mask_path = os.path.join(output_folder_seg, f"mask_{filename.replace('.jpg', f'_{idx}.png')}")
                mask_img = Image.fromarray(mask)  # Convert to binary mask
                mask_img.save(mask_path)

                print(f"✅ Mask saved: {mask_path}")

            except Exception as e:
                print(f"❌ SAM Error on {filename}, bbox {idx}: {e}")
                continue

        end_time = time.time()
        print(f"✅ SAM Worker {thread_name} Finished Processing: {filename} in {end_time - start_time:.3f} sec")


# ========== 🚀 Multi-threading Execution ==========
if __name__ == "__main__":
    input_queue = Queue()
    total_start_time = time.time()  # Start global timer

    # Start YOLO Thread
    yolo_thread = Thread(target=run_yolo, args=(input_queue,))
    yolo_thread.start()

    # Start SAM Threads and track their completion
    sam_threads = []
    with ThreadPoolExecutor(max_workers=NUM_SAM_WORKERS) as executor:
        futures = [executor.submit(run_sam, input_queue) for _ in range(NUM_SAM_WORKERS)]

        # Ensure all SAM workers finish before moving on
        for future in as_completed(futures):
            try:
                future.result()  # This ensures that any errors in SAM processing are raised
            except Exception as e:
                print(f"❌ SAM Worker Error: {e}")

    # Wait for YOLO to finish
    yolo_thread.join()

    total_end_time = time.time()  # End global timer
    total_time = total_end_time - total_start_time
    fps = 101 / total_time  # Assuming 101 images processed

    print(f"\n✅ All images successfully processed in {total_time:.2f} sec 🔥 FPS: {fps:.2f}")
