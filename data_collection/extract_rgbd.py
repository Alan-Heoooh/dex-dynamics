import os
import time
import subprocess

# Path to your script that extracts color and depth images
EXTRACTION_SCRIPT = "/home/coolbot/Documents/git/dex-dynamics/data_collection/sync_ros_to_data.py"

# List of ROS bag files to play
# BAG_FILES = [
#     "/home/coolbot/data/hand_object_perception/ros_record/scene_0003/scene_0003_0.db3",
#     "/home/coolbot/data/hand_object_perception/ros_record/scene_0001/scene_0001_0.db3",
#     "/home/coolbot/data/hand_object_perception/ros_record/scene_0002/scene_0002_0.db3",
# ]

start_idx= 100
end_idx = 109

BAG_FILES = [f"/home/coolbot/data/hand_object_perception/ros/scene_{i:04d}/scene_{i:04d}_0.db3" for i in range(start_idx, end_idx+1)]

# Base directory for saving images
BASE_SAVE_DIR = "/home/coolbot/data/hand_object_perception/train"

def run_extraction_script(save_dir):
    """Runs the image extraction script with a specific save directory."""
    print(f"Starting image extraction with save directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)  # Ensure the directory exists
    process = subprocess.Popen(["python3", EXTRACTION_SCRIPT, save_dir])
    time.sleep(5)  # Give it some time to initialize
    return process

def play_ros_bag(bag_path):
    """Plays a single ROS bag file."""
    print(f"Playing bag: {bag_path}")
    subprocess.run(["ros2", "bag", "play", bag_path], check=True)

def main():
    for bag_path in BAG_FILES:
        # Extract bag file name (without extension) to create a unique directory
        bag_name = os.path.basename(bag_path).replace(".db3", "")
        save_dir = os.path.join(BASE_SAVE_DIR, bag_name)

        # Start the extraction script for this specific bag
        extraction_process = run_extraction_script(save_dir)

        try:
            # Play the current bag
            play_ros_bag(bag_path)
            time.sleep(2)  # Small delay before playing the next bag
        finally:
            # Stop the extraction script after processing the bag
            print(f"Stopping image extraction for {bag_name}...")
            extraction_process.terminate()
            extraction_process.wait()

if __name__ == "__main__":
    main()