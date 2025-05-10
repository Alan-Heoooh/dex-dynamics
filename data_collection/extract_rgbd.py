import os
import time
import subprocess

# Path to your script that extracts color and depth images
EXTRACTION_SCRIPT = "data_collection/sync_ros_to_data.py"

# List of ROS bag files to play


start_idx = 600
end_idx = 759

BAG_FILES = [f"/media/coolbot/Extreme Pro/data/ros_record/scene_{i:04d}/scene_{i:04d}_0.db3" for i in range(start_idx, end_idx+1)]

# Base directory for saving images
BASE_SAVE_DIR = "/media/coolbot/Extreme Pro/data/train_0413_thumb_press+thumb_pinch"

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
        if not os.path.exists(bag_path):
            print(f"Bag file {bag_path} does not exist. Skipping...")
            continue
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