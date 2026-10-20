import subprocess
import time
from datetime import datetime, timedelta
import os
import portalocker
import shutil

# def record_stream(rtmp_url, output_folder, segment_length):
#     # 生成文件名中的时间戳
#     timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
#     output_template = os.path.join(output_folder, f"live_segment_{timestamp}_%03d.mp4")
#     # 构建 ffmpeg 命令
#     command = [
#         'ffmpeg', '-rw_timeout', '5000000', '-i', rtmp_url, '-c', 'copy', '-map', '0', '-segment_time', 
#         str(segment_length), '-f', 'segment', '-reset_timestamps', '1', output_template
#     ]
#     subprocess.run(command)

def streamCaptureRTMP(rtmp_url, rootFolder, video_direction, shutdown_event, outputFile, segment_duration):
    # rtmp_url = "rtmp://131.247.15.242/live/drone01"
    output_folder = rootFolder+'streamingFolder/'
    watch_folder = rootFolder+'watchFolder/'
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(watch_folder, exist_ok=True)
    with open(outputFile, 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('Stream capture process started!\n')
        portalocker.unlock(file)

    # 检测直播是否在进行的逻辑在这里可能需要更复杂的方法
    # 以下是一个简单的循环，需要根据实际情况调整
    while not shutdown_event.is_set():
        try:
            print("Checking live stream...")
            # 尝试记录流，如果直播没有进行，ffmpeg命令会失败
            # record_stream(rtmp_url, output_folder, 120)
            # Start recording
            # 生成文件名中的时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # output_template = os.path.join(output_folder, f"{timestamp}-{video_direction}-%03d.mp4")
            # # 构建 ffmpeg 命令
            # command = [
            #     'ffmpeg', '-rw_timeout', '5000000', '-i', rtmp_url, '-c', 'copy', '-map', '0', '-segment_time', 
            #     str(segment_duration), '-f', 'segment', '-reset_timestamps', '1', output_template
            # ]
            output_filename = os.path.join(output_folder, f"{timestamp}-{video_direction}.mp4")
            command = ['ffmpeg', '-rw_timeout', '5000000', '-i', rtmp_url, '-c', 'copy', '-map', '0', output_filename]
            with open(outputFile, 'a') as file:
                portalocker.lock(file, portalocker.LOCK_EX)
                file.write(f'Stream capturing: {command}\n')
                portalocker.unlock(file)
            # subprocess.run(command)
            # 确保stdin被设置为PIPE以发送命令给ffmpeg
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            # 记录开始时间
            start_time = datetime.now()
            # 计算结束时间
            end_time = start_time + timedelta(seconds=segment_duration)
            # 检查是否达到指定的录制时长
            while datetime.now() < end_time:
                time_diff = (datetime.now() - start_time).total_seconds()
                print(f'Stream captured: {time_diff}\n')
                with open(outputFile, 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write(f'Stream captured: {time_diff}\n')
                    portalocker.unlock(file)
                # If main process stops me
                if(shutdown_event.is_set()):
                    print("shutdown_event is set! Existing...")
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write("shutdown_event is set! Existing...\n")
                        portalocker.unlock(file)
                    break
                if process.poll() is not None:
                    # FFmpeg 进程已经结束
                    print("FFmpeg process ended.")
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write("FFmpeg process ended.\n")
                        portalocker.unlock(file)
                    break
                # If any error occurs
                output = process.stderr.readline()
                if output:
                    print("FFMPEG output: " + output)
                    if "error" in output.lower():
                        print("FFmpeg error detected:", output.strip())
                        with open(outputFile, 'a') as file:
                            portalocker.lock(file, portalocker.LOCK_EX)
                            file.write(f'FFmpeg error detected: {output.strip()}\n')
                            portalocker.unlock(file)
                        break
                time.sleep(1)
                # pass
            # 时间到，发送'q'命令给ffmpeg进行优雅地退出
            if process.poll() is None:
                process.stdin.write('q')
                process.stdin.flush()
            try:
                process.communicate(timeout=5)
                if(os.path.exists(output_filename)):
                    shutil.move(output_filename, watch_folder)
                    with open(outputFile, 'a') as file:
                        portalocker.lock(file, portalocker.LOCK_EX)
                        file.write(f"Stream captured: {output_filename.split('/')[-1]}\n")
                        portalocker.unlock(file)
            except subprocess.TimeoutExpired:
                print("FFmpeg did not exit in time. Forcing termination.")
                process.terminate()  # Force terminate
                process.wait()  # Ensure process resources are cleaned up
            
        except subprocess.CalledProcessError as e:
            print("Live stream not active. Retrying in 60 seconds.")
            with open(outputFile, 'a') as file:
                portalocker.lock(file, portalocker.LOCK_EX)
                file.write(f'Live stream not active. Retrying in 60 seconds...\n')
                portalocker.unlock(file)
            time.sleep(5)


# if __name__ == "__main__":
#     shutdown_event = Event()
#     streamCaptureRTMP("rtmp://131.247.15.242/live/drone01", "C:/yoloV4/thermal_90_degrees_phase1/darknet/test_videos/Test/videos/realtimeProcessing/", 
#                       "Omission", shutdown_event, outputFile, segment_duration)





# import av
# import os
# import time
# from datetime import datetime

# rtmp_url = "rtmp://131.247.15.242/live/drone01"
# save_path = "C:/yoloV4/thermal_90_degrees_phase1/darknet/test_videos/Test/videos/LiveStreaming/streamingFolder"
# retry_interval = 5  # Seconds before trying to reconnect
# segment_duration = 60  # Duration of each video segment in seconds

# def generate_filename(prefix, suffix=".mp4"):
#     """Generate a filename with the current datetime."""
#     now = datetime.now()
#     date_time = now.strftime("%y%m%d_%H%M%S")
#     return f"{prefix}{date_time}{suffix}"

# def capture_stream_segmented(rtmp_url, save_path, segment_duration):
#     next_segment_time = time.time() + segment_duration
#     output_stream = None

#     while True:
#         output_filename = generate_filename("", "-omission.mp4")
#         output_filepath = os.path.join(save_path, output_filename)

#         try:
#             input_stream = av.open(rtmp_url)
#             video_stream = next(s for s in input_stream.streams if s.type == 'video')
#             output_stream = av.open(output_filepath, mode='w')
#             output_container = output_stream.add_stream(template=video_stream)

#             first_pts = None
#             print(f"Segment saving: {output_filename}")
#             for packet in input_stream.demux(video_stream):
#                 if time.time() >= next_segment_time:
#                     break
#                 if packet.dts is not None:  # Ensure proper decoding
#                     # Reset timestamps for each segment to start from 0
#                     if first_pts is None:
#                         first_pts = packet.pts
#                     packet.pts -= first_pts
#                     packet.dts -= first_pts
#                     packet.stream = output_container
#                     output_stream.mux(packet)

#             # Close the current file and start a new one
#             output_stream.close()
#             print(f"Segment saved: {output_filename}")
#             if time.time() >= next_segment_time:
#                 next_segment_time = time.time() + segment_duration

#         except av.AVError as e:
#             print(f"Error occurred: {e}")
#             if output_stream is not None:
#                 output_stream.close()
#             time.sleep(retry_interval)
#             continue

#         except KeyboardInterrupt:
#             print("Exiting...")
#             if output_stream is not None:
#                 output_stream.close()
#             break

#         finally:
#             if output_stream is not None:
#                 output_stream.close()

# if not os.path.exists(save_path):
#     os.makedirs(save_path)

# capture_stream_segmented(rtmp_url, save_path, segment_duration)




# from multiprocessing import Process, Event
# import av
# import os
# import time
# from datetime import datetime

# rtmp_url = "rtmp://131.247.15.242/live/drone01"
# save_path = "C:/yoloV4/thermal_90_degrees_phase1/darknet/test_videos/Test/videos/LiveStreaming/streamingFolder"
# retry_interval = 5  # Seconds before trying to reconnect
# segment_duration = 60  # Duration of each video segment in seconds

# def generate_filename(prefix, suffix=".mp4"):
#     now = datetime.now()
#     date_time = now.strftime("%y%m%d_%H%M%S")
#     return f"{prefix}{date_time}{suffix}"

# def capture_stream_segment(rtmp_url, save_path, output_filename, segment_duration, connected, stop_event):
#     try:
#         output_stream = None
#         input_stream = av.open(rtmp_url)
#         video_stream = next(s for s in input_stream.streams if s.type == 'video')
#         output_stream = av.open(os.path.join(save_path, output_filename), mode='w')
#         output_container = output_stream.add_stream(template=video_stream)

#         first_pts = None
#         # streamCnt = 0  # Flag to check if we received any packet during the loop
#         connected.set()
#         next_segment_time = time.time() + segment_duration
#         print(f"Segment saving: {output_filename}")
#         for packet in input_stream.demux(video_stream):
#             if stop_event.is_set():
#                 break
#             if time.time() >= next_segment_time:
#                 break
#             if packet.dts is not None:  # Ensure proper decoding
#                 # Reset timestamps for each segment to start from 0
#                 if first_pts is None:
#                     first_pts = packet.pts
#                 packet.pts -= first_pts
#                 packet.dts -= first_pts
#                 packet.stream = output_container
#                 output_stream.mux(packet)

#         # Close the current file and start a new one
#         output_stream.close()
#         print(f"Segment saved: {output_filename}")
#         connected.clear()
#         # if time.time() >= next_segment_time:
#         #     next_segment_time = time.time() + segment_duration

#     except Exception as e:
#         print(f"Error occurred: {e}")
#         if output_stream is not None:
#             output_stream.close()

#     finally:
#         print('Finally Part!')
#         connected.clear()
#         if output_stream is not None:
#             print(f"Segment saved: {output_filename}")
#             output_stream.close()

# def check_live(rtmp_url, check_live_event):
#     try:
#         input_stream = av.open(rtmp_url)
#         video_stream = next(s for s in input_stream.streams if s.type == 'video')
#         print("check_live_event set!")
#         check_live_event.set()
#         return
        
#     except Exception as e:
#         print(f"Error occurred: {e}")
#         return

# def capture_stream_segmented(rtmp_url, save_path, segment_duration):
#     try:
#         liveStreamFlag = True
#         connected = Event()
#         stop_event = Event()
#         check_live_event = Event()
#         process = None
#         processCheck = None
#         while liveStreamFlag:
#             check_fail_cnt = 0
#             connected.clear()
#             stop_event.clear()
#             check_live_event.clear()
#             output_filename = generate_filename("", "-omission.mp4")
#             process = Process(target=capture_stream_segment, args=(rtmp_url, save_path, output_filename, segment_duration, connected, stop_event))
#             process.start()
#             process.join(timeout=retry_interval)
#             if not connected.is_set():
#                 print("Connection timeout or failed. Trying to reconnect...")
#                 process.terminate()  # Attempt to terminate the process
#                 process.join()  # Clean up
#                 time.sleep(retry_interval)
#                 continue
#             else:
#                 # process.join()
#                 while liveStreamFlag:
#                     check_live_event.clear()
#                     processCheck = Process(target=check_live, args=(rtmp_url, check_live_event))
#                     processCheck.start()
#                     processCheck.join(timeout=retry_interval)
#                     print(f"Checking Live...{check_fail_cnt}")
#                     if not connected.is_set():
#                         if process is not None:
#                             process.terminate()
#                             process.join()
#                         if processCheck is not None:
#                             processCheck.terminate()
#                             processCheck.join()
#                         break
#                     if not check_live_event.is_set():
#                         check_fail_cnt += 1
#                         print(f"Checking Live...{check_fail_cnt}")
#                         if check_fail_cnt > 2:
#                             # Live stops
#                             liveStreamFlag = False
#                             stop_event.set()
#                             time.sleep(1)
#                             if process is not None:
#                                 process.terminate()
#                                 process.join()
#                             if processCheck is not None:
#                                 processCheck.terminate()
#                                 processCheck.join()
#                             break
#                     else:
#                         check_fail_cnt = 0
#     except Exception as e:
#         print(f'Error: {e}')
#         # if process is not None:
#         #     stop_event.set()
#         #     time.sleep(2)
#         #     process.terminate()
#         #     process.join()  # Clean up
#     finally:
#         if process is not None:
#             process.terminate()
#             process.join()
#         if processCheck is not None:
#             processCheck.terminate()
#             processCheck.join()
            
# if __name__ == '__main__':

#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     capture_stream_segmented(rtmp_url, save_path, segment_duration)

