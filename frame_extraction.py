import numpy as np
from rosbags.highlevel import AnyReader
import cv2
import os
from pathlib import Path
import sys

def frame_extraction(rosbag_path: str, topic_name: str, output_dir: str, total_frames: int) -> int:

    bag_path = Path(rosbag_path)
    frames_dir = Path(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)  # Create frames directory and parents if they don't exist
    frame_count = total_frames + 1 

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == topic_name]

         # Initialize frame counter

        for connection, timestamp, rawdata in reader.messages(connections=connections):
            msg = reader.deserialize(rawdata, connection.msgtype)

            # Check if this is a compressed image
            if hasattr(msg, "format"):  # sensor_msgs/CompressedImage
                np_arr = np.frombuffer(msg.data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
            else:  # sensor_msgs/Image (raw)
                # Convert raw bytes to a numpy array
                # reshape: height x width x channels
                if 'rgb8' in msg.encoding or 'bgr8' in msg.encoding:
                    dtype = np.uint8
                    channels = 3
                elif 'mono8' in msg.encoding:
                    dtype = np.uint8
                    channels = 1
                elif "16UC1" in msg.encoding or "mono16" in msg.encoding:
                    dtype = np.uint16
                    channels = 1
                else:
                    raise ValueError(f"Unsupported encoding: {msg.encoding}")

                frame = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, channels)

                # If mono, convert to BGR for VideoWriter
                if channels == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    
            # Save the frame as a PNG file with sequential numbering
            frame_path = frames_dir / f"frame_{frame_count:04d}.png" if "depth" not in topic_name else frames_dir / f"depth_frame{frame_count:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1
            
    return frame_count

def main(year: int):
    rosbag_paths = [
    "comp2.2025-05-17-10-1.bag",
    "comp2.2025-05-17-10.bag",
    "comp3.2025-05-17-10-40.bag",
    "comp3.2025-05-17-9-40.bag",
    "comp2.2025-05-16-10-1.bag",
    "comp2.2025-05-16-10-1.bag",
    "comp3.2025-05-17-10-40.bag",
    "comp2.2025-05-16-10-1.bag",
    "comp3.2025-05-17-9-40.bag",
    "comp2.2025-05-16-10-1.bag",
    "comp3.2025-05-17-10-40.bag",
    "comp2.2025-05-16-10-1.bag",
] if year == 2025 else [
    "comp1_manual_exdep_2024-05-11-13-26-12.bag",
    "comp3_auto_2024-05-12-11-02-23.bag",
    "comp4_auto_dep_2024-05-15-16-22-28.bag",
    "comp4_exc_2024-05-15-16-28-05.bag",
    "comp4_setup_2024-05-15-15-52-04.bag",
    "exp_2024-05-15-16-22-43.bag",
    "ksc_pits_2024-05-15-15-02-42.bag",
    "ksc_pits_2024-05-15-15-49-43.bag",
    "ksc_pits_2024-05-15-15-53-42.bag",
    "ksc_pits_2024-05-15-16-00-06.bag",
    "ksc_pits_2024-05-15-16-27-55.bag",
    "ksc_pits_2024-05-15-16-52-10.bag",
    "ksc_pits_driving_2024-05-15-15-52-05.bag",
    "ksc_pits_homing_2024-05-15-16-33-22.bag",
    "ucf_pits_2024-05-12-09-02-38.bag",
    "ucf_pits_2024-05-12-09-41-32.bag",
    "ucf_pits_2024-05-12-11-13-05.bag",
    "ucf_pits_2024-05-12-11-22-07.bag",
    "ucf_pits_driving_2024-05-12-10-08-36.bag",
    "ucf_pits_driving_2024-05-12-10-24-52.bag",
    "comp1_auto_2024-05-11-13-13-17.bag",
    "comp2_auto_2024-05-12-11-06-08.bag",
    "comp3_dep_2024-05-12-11-16-12.bag",
    "comp4_auto_exc_2024-05-15-16-16-48.bag",
    "comp4_manual_exdep_2024-05-15-16-23-31.bag",
    "comp5_auto_2024-05-15-15-50-49.bag",
    "exp_2024-05-15-16-55-26.bag",
    "ksc_pits_2024-05-15-15-48-43.bag",
    "ksc_pits_2024-05-15-15-50-22.bag",
    "ksc_pits_2024-05-15-15-53-45.bag",
    "ksc_pits_2024-05-15-16-17-49.bag",
    "ksc_pits_2024-05-15-16-32-33.bag",
    "ksc_pits_2024-05-15-16-53-01.bag",
    "ksc_pits_homing_2024-05-15-16-23-53.bag",
    "ucf_pits_2024-05-11-18-16-49.bag",
    "ucf_pits_2024-05-12-09-03-50.bag",
    "ucf_pits_2024-05-12-09-47-09.bag",
    "ucf_pits_2024-05-12-11-17-42.bag",
    "ucf_pits_2024-05-12-11-23-43.bag",
    "ucf_pits_driving_2024-05-12-10-13-25.bag",
    "comp1_manual_2024-05-11-13-14-35.bag",
    "comp2_manual_2024-05-12-11-18-02.bag",
    "comp3_exc_2024-05-12-11-14-02.bag",
    "comp4_auto2_2024-05-15-16-10-37.bag",
    "comp4_setup_2024-05-15-15-49-40.bag",
    "comp5_setup_2024-05-15-15-49-59.bag",
    "ksc_pits_2024-05-15-14-38-17.bag",
    "ksc_pits_2024-05-15-15-48-52.bag",
    "ksc_pits_2024-05-15-15-51-20.bag",
    "ksc_pits_2024-05-15-15-54-05.bag",
    "ksc_pits_2024-05-15-16-19-38.bag",
    "ksc_pits_2024-05-15-16-34-56.bag",
    "ksc_pits_2024-05-15-16-54-10.bag",
    "ksc_pits_homing_2024-05-15-16-25-02.bag",
    "ucf_pits_2024-05-11-18-23-48.bag",
    "ucf_pits_2024-05-12-09-06-31.bag",
    "ucf_pits_2024-05-12-10-00-11.bag",
    "ucf_pits_2024-05-12-11-19-18.bag",
    "ucf_pits_2024-05-12-11-34-57.bag",
    "ucf_pits_driving_2024-05-12-10-17-34.bag",
    "comp1_manual_dep_2024-05-11-13-15-50.bag",
    "comp2_setup_2024-05-12-11-00-59.bag",
    "comp4_auto_2024-05-15-15-56-27.bag",
    "comp4_ending_2024-05-15-16-30-54.bag",
    "comp4_setup_2024-05-15-15-51-13.bag",
    "exp_2024-05-15-16-15-08.bag",
    "ksc_pits_2024-05-15-14-49-00.bag",
    "ksc_pits_2024-05-15-15-48-53.bag",
    "ksc_pits_2024-05-15-15-52-12.bag",
    "ksc_pits_2024-05-15-15-56-39.bag",
    "ksc_pits_2024-05-15-16-22-18.bag",
    "ksc_pits_2024-05-15-16-45-30.bag",
    "ksc_pits_auto_test_2024-05-15-16-30-35.bag",
    "ksc_pits_homing_2024-05-15-16-26-03.bag",
    "ucf_pits_2024-05-12-08-45-17.bag",
    "ucf_pits_2024-05-12-09-08-34.bag",
    "ucf_pits_2024-05-12-11-03-07.bag",
    "ucf_pits_2024-05-12-11-21-49.bag",
    "ucf_pits_2024-05-12-12-07-01.bag",
    "ucf_pits_driving_2024-05-12-10-22-01.bag"
]
    topic_names = ["/d455_front/camera/aligned_depth_to_color/image_raw", '/d455_front/camera/color/image_rect_color', "/d455_back/camera/aligned_depth_to_color/image_raw", '/d455_back/camera/color/image_rect_color'] if year == 2025 else ["/d455_back/camera/color/image_raw", "/d455_back/camera/aligned_depth_to_color/image_raw"]
    output_dir = './frames'
    # The lines above should be handled by the command line arguments
    total_frames = 0
    
    for bag_path in rosbag_paths:
        output_dir_ = os.path.join(output_dir, Path(bag_path).stem)
        for i in range(4):
            if i < 2:
                frames_extracted = frame_extraction(bag_path, topic_names[i], output_dir_, total_frames)
                total_frames += frames_extracted + 1
            else:
                frames_extracted = frame_extraction(bag_path, topic_names[i], output_dir_, total_frames)
                total_frames += frames_extracted + 1

if __name__ == "__main__":
    year = sys.argv[1]
    
    main(year)
    