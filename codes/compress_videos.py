import subprocess

def compress_video(input_file, output_file, bitrate="500k"):
    command = f'ffmpeg -i "{input_file}" -vcodec libx264 -b:v {bitrate} -crf 28 "{output_file}"'
    subprocess.run(command, shell=True)
    print(f"🎥 Compressed video saved: {output_file}")

if __name__ == "__main__":
    compress_video("output/sample.mp4", "output/sample_compressed.mp4")