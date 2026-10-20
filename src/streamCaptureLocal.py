import os
import time
import shutil
import portalocker
from datetime import datetime


def streamCaptureLocal(video_path, rootFolder, video_direction, shutdown_event, outputFile, segment_duration=120):
    """Simulates live stream capture by copying a local video file to a watchFolder.

    For testing: copies the whole video with its original filename so that
    pre-computed darknet results can be matched. For future edge-device use,
    this can be swapped back to segment-based capture.
    """
    watch_folder_1 = rootFolder + 'watchFolder1/'
    os.makedirs(watch_folder_1, exist_ok=True)

    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write(f'Local stream capture started: {video_path}\n')
        portalocker.unlock(file)

    if not os.path.exists(video_path):
        with open(outputFile, 'a') as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.write(f'Error: Video file not found: {video_path}\n')
            portalocker.unlock(file)
        return

    # Copy the video to watchFolder1 with its original filename
    original_filename = os.path.basename(video_path)
    dest_path = os.path.join(watch_folder_1, original_filename)
    shutil.copy2(video_path, dest_path)

    print(f'Video copied to: {dest_path}')
    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write(f'Video copied: {original_filename} -> watchFolder1\n')
        portalocker.unlock(file)

    # Wait until shutdown so the process stays alive for status tracking
    while not shutdown_event.is_set():
        time.sleep(2)

    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write(f'Local stream capture finished.\n')
        portalocker.unlock(file)
