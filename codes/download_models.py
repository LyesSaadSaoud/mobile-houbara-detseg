import os

def download_model(url, save_path):
    import requests
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024):
            f.write(chunk)

if not os.path.exists("models"):
    os.makedirs("models")

print("Downloading YOLO model...")
download_model("https://your-model-link.com/yolo10.pt", "models/best_yolo10_40k.pt")

print("Downloading MobileSAM model...")
download_model("https://your-model-link.com/mobilesam.pt", "models/mobile_sam.pt")

print("✅ Models downloaded!")