import os
# Set API keys as environment variables before running (see start.sh)
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
# os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
# os.environ["GEMINI_API_KEY"] = "your-gemini-key"
from flask import Flask, redirect, url_for, render_template, request, jsonify, Response, send_from_directory, abort
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from flask_bootstrap import Bootstrap5
from flask_socketio import SocketIO, join_room
import time
import subprocess
from multiprocessing import Process, Pipe, Event
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
import portalocker
from bailiAPLibWeb import autoProcessing
from streamCaptureYouTube import streamCaptureYouTube
from streamCaptureRTMP import streamCaptureRTMP
from streamCaptureTwitch import streamCaptureTwitch
from streamCaptureLocal import streamCaptureLocal
from streamCaptureCamera import streamCaptureCamera
from vlm_analysis import analyze_incident, save_results as save_vlm_results, VLM_CONFIG
from realtime_detector import RealtimeProcessor
import json
import os, sys
import threading
from datetime import datetime, timedelta
import pandas as pd
from zoneinfo import ZoneInfo
import math
import base64
import socket
from flask_talisman import Talisman
from flask import g
import cv2 as cv

app = Flask(__name__)
bootstrap=Bootstrap5(app)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# Variables set up — edge device paths (Linux/Jetson)
_project_dir = os.path.dirname(os.path.abspath(__file__))
rootFolder = os.path.join(_project_dir, "dashboard_results") + "/"
os.makedirs(rootFolder, exist_ok=True)
experimentFolder = None
subFolders = ['.cacheFiles/', 'recordsFolder/', 'processedFolder/', 'streamingFolder/', 'watchFolder1/', 'watchFolder2/', 'watchFolder3/']
outputFiles = {}
outputFilesName = ['streamCaptureLogs', 'videoProcessingLogs1', 'videoProcessingLogs2', 'videoProcessingLogs3', 'gpsFetchResults', 'detectionResultsLogs', 'mainOutput']
detectionResultsFile = None

processHandler = {'stream': None, 'video1': None, 'video2': None, 'video3': None}
# videoProcessingParentPipe = None
# videoProcessingChildPipe = None
# youtube_url = None
video_duration = 120
drone_direction = None
highwayName = None
video_direction = None
rtmp_url = "rtmp://131.247.15.242/live/drone01"
# Remote dashboard URL — edge device sends detection results here for VLM analysis
# Set to None if running standalone (no remote dashboard)
dashboard_url = os.environ.get('DASHBOARD_URL', None)  # e.g., "http://192.168.1.100:4430"

# Video source configuration for edge device:
#   - Set to a file path for testing with local video
#   - Set to "csi" for Jetson CSI camera
#   - Set to 0 or 1 for USB camera (e.g., thermal USB camera)
#   - Set to "rtsp://..." for RTSP stream
#   - Set to None to use RTMP stream (streamCaptureTwitch)
local_video_path = None  # Set to file path for local video, None for live camera
camera_source = 0  # HDMI capture: 0 or 1. Set to None for local video mode
detectionOperationFlags = {'start': False, 'stop': False, 'pause': False}
# Statue variables for status of the detection program and each process
# statusDict = {'detectionStatus': False, 'pauseStatus': False, 'streamCaptureStatus': False, 'videoProcessingStatus': False, 'gpsFetchStatus': False,
#                'youtubeUrl': youtube_url, 'droneDirection': drone_direction, 'videoDuration': video_duration, 'highwayName': highwayName}
statusDict = {'detectionStatus': False, 'pauseStatus': False, 'streamCaptureStatus': False, 'videoProcessingStatus': False, 'gpsFetchStatus': False,
                'droneDirection': drone_direction, 'videoDuration': video_duration, 'highwayName': highwayName}

shutdownEvents = {'stream': Event(), 'video': Event(), 'thread': threading.Event()}

# VLM analysis state
vlm_report_status = {'status': 'idle', 'results': None, 'error': None}

# Real-time processor instance
realtime_proc = None
OUTRI_DARTS_GATE_STATUS = "/tmp/outri_darts_gate_status.json"

def latest_outri_gate_status():
    try:
        with open(OUTRI_DARTS_GATE_STATUS, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return None

# Flask-Login配置
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'  # 未认证用户将被重定向到的登录视图的名称

# 假设的用户存储
users = {'admin': {'password': 'usfsumlab'}}

def require_outri_dashboard_clearance():
    """Optional second factor: require local Outri clearance before dashboard login."""
    if os.environ.get("OUTRI_DASHBOARD_GATE", "0").lower() not in ("1", "true", "yes", "on"):
        return True, "Outri dashboard gate disabled"

    sock_path = os.environ.get("OUTRI_GATE_SOCKET", "/tmp/outri_gate.sock")
    payload = {
        "module_id": os.environ.get("OUTRI_DASHBOARD_MODULE", "module-beta-v2"),
        "action": os.environ.get("OUTRI_DASHBOARD_ACTION", "DARTS dashboard access"),
        "command": "dashboard-login",
    }
    try:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.settimeout(2.0)
        client.connect(sock_path)
        with client:
            client.sendall((json.dumps(payload) + "\n").encode("utf-8"))
            response = json.loads(client.makefile().readline() or "{}")
        return response.get("status") == "ALLOW", response.get("reason", "Outri denied dashboard access")
    except Exception as exc:
        return False, f"Outri gate unavailable: {exc}"

# 用户类
class User(UserMixin):
    def __init__(self, id):
        self.id = id

# 用户加载函数
@login_manager.user_loader
def load_user(user_id):
    if user_id in users:
        return User(user_id)
    return None

@app.route('/')
def home():
    # 如果用户已经登录，则重定向到detection路由
    if current_user.is_authenticated:
        return redirect(url_for('detection'))
    # 否则，重定向到登录页面
    return redirect(url_for('login'))

# 登录路由
@app.route('/login', methods=['GET', 'POST'])
def login():
    # 检查用户是否已经登录
    if current_user.is_authenticated:
        # 如果已经登录，重定向到detection页面
        return redirect(url_for('detection'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username in users and users[username]['password'] == password:
            allowed, reason = require_outri_dashboard_clearance()
            if not allowed:
                return f'Outri clearance required: {reason}', 403
            user = User(username)
            login_user(user)
            return redirect(url_for('detection'))
        else:
            return 'Invalid username or password'
    return render_template('login.html', active_page='login')

# 受保护的路由
@app.route('/protected')
@login_required
def protected():
    return 'Logged in as: ' + UserMixin.get_id(current_user)  # 使用current_user获取当前登录的用户

# 登出路由
@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))



# socketio setup
@socketio.on('connect')
def handle_connect():
    print('From app.py: Client connected')
    # socketio.emit('updateStatus', statusDict)

@socketio.on('join')
def on_join(data):
    room = data['room']
    join_room(room)
    if(room=='detection'):
        socketio.emit('updateStatus', statusDict, room=room)
        # if detection results exist, show that.
        if(detectionResultsFile):
            detectionResultsFileContent = None
            with open(detectionResultsFile, 'r') as file:
                portalocker.lock(file, portalocker.LOCK_SH)
                detectionResultsFileContent = file.read()
                portalocker.unlock(file)
            if(detectionResultsFileContent):
                socketio.emit('detectionResults', {'message': detectionResultsFileContent}, room='detection')
    elif(room=='logs'):
        # print(outputFiles)
        if outputFiles:
            for key, value in outputFiles.items():
                # socketio.emit('emitOutputFiles', {'message': f"Checking file of {key}"}, room='logs')
                # print(key, value)
                if(os.path.exists(value) and (not key=='detectionResults')):
                    # print(f"{key} exists!\n")
                    with open(value, 'r') as file:
                        # print(f"{key} reading!\n")
                        portalocker.lock(file, portalocker.LOCK_SH)
                        fileContent = file.read()
                        portalocker.unlock(file)
                    socketio.emit(key, {'message': fileContent}, room=room)

@socketio.on('disconnect')
def handle_disconnect():
    print('From app.py: Client disconnected')

# Certificate Steps!
# @app.route('/.well-known/pki-validation/<filename>')
# def well_known_pki_validation(filename):
#     # 提供文件的目录路径
#     directory = 'static/'
#     # 使用 send_from_directory 安全地发送文件内容
#     return send_from_directory(directory, filename, as_attachment=False)

# Router set up
@app.route('/detection')
@login_required
def detection():
    return render_template('detection.html', active_page='detection', camera_source=camera_source, local_video_path=local_video_path)

@app.route('/logs')
@login_required
def logs():
    return render_template('logs.html', active_page='logs')

@app.route('/history')
@login_required
def history():
    return render_template('history.html', active_page='history')

@app.route('/gps-upload', methods=['POST'])
def receive_gps_data():
    auth = request.headers.get('Authorization')
    if auth:
        auth_type, auth_info = auth.split(None, 1)
        if auth_type.lower() == 'basic':
            username, password = base64.b64decode(auth_info).decode().split(':', 1)
            # 在这里验证用户名和密码
            if username != "admin" or password != "usfsumlab":
                # return "You are logged in.", 200
                return Response("Access Denied", status=403)
        else:
            return Response("Access Denied", status=403)
    else:
        return Response("Access Denied", status=403)
    data = request.data.decode('utf-8')
    # print(request.headers.get('Content-Type'))
    print(data)
    if(not data.isascii()):
        return 'Data received\n', 200
    if((len(data)<35) or (len(data)>55)):
        return 'Data received\n', 200
    # dataSlices = data.split(',')
    # if(len(dataSlices)!=9):
    #     return 'Data received\n', 200
    # if((not dataSlices[1]=='N') and (not dataSlices[1]=='S')):
    #     return 'Data received\n', 200
    # if((not dataSlices[3]=='W') and (not dataSlices[3]=='E')):
    #     return 'Data received\n', 200
    # if(('.' not in dataSlices[0]) and ('.' not in dataSlices[2])):
    #     return 'Data received\n', 200
    # # Begin processing
    # # if (not shutdownEvents['video'].is_set() and outputFiles):
    # # print('Begin to process...')
    # latDD = dataSlices[0][0:2]
    # latMM = dataSlices[0][2:]
    # latitudeRes = int(latDD) + ((float(latMM))/60)
    # if(dataSlices[1]=='S'):
    #     latitudeRes = -latitudeRes
    # lonDD = dataSlices[2][0:3]
    # lonMM = dataSlices[2][3:]
    # longitudeRes = int(lonDD) + (float(lonMM)/60)
    # if(dataSlices[3]=='W'):
    #     longitudeRes = -longitudeRes
    # datetimeStr = dataSlices[4] + '_' + dataSlices[5].split('.')[0]
    # datetimeRes = datetime.strptime(datetimeStr, '%d%m%y_%H%M%S')
    # # 首先将时间本地化为UTC（如果原始时间代表的是UTC时间），然后转换到"America/New_York"时区
    # datetimeRes = datetimeRes.replace(tzinfo=ZoneInfo("UTC")).astimezone(ZoneInfo("America/New_York"))
    # datetimeStr = datetime.strftime(datetimeRes, '%Y%m%d_%H%M%S')
    # # finalStr = datetimeStr + ',' + str(latitudeRes) + ',' + str(longitudeRes)
    # finalStr = datetimeStr + f",{latitudeRes:.9f}" + f",{longitudeRes:.9f}"
    # # print(f"Final result to save in a csv file is: {finalStr}\n")
    if (not shutdownEvents['video'].is_set() and outputFiles):
        with open(outputFiles['gpsFetchResults'], 'a') as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.write(data + '\n')
            portalocker.unlock(file)
    return 'Data received\n', 200
    # if (not shutdownEvents['video'].is_set() and outputFiles):
    #     data = request.data
    #     if (data.isascii() and (len(data.decode('utf-8').split(','))>=3)):  # 简单校验，确保接收到的是纯文本数据
    #         with open(outputFiles['gpsFetchResults'], 'a') as file:
    #             portalocker.lock(file, portalocker.LOCK_EX)
    #             file.write(data.decode('utf-8') + '\n')
    #             portalocker.unlock(file)
    #         return 'Data received\n', 200
    #     else:
    #         return 'Invalid data format\n', 400

@app.route('/start-detection', methods=['POST'])
@login_required
def start_detection():
    # global processHandler, statusDict, youtube_url, drone_direction, highwayName, video_direction, video_duration, \
    #     detectionOperationFlags, experimentFolder, outputFiles, detectionResultsFile
    global processHandler, statusDict, drone_direction, highwayName, video_direction, video_duration, \
        detectionOperationFlags, experimentFolder, outputFiles, detectionResultsFile
    if(detectionOperationFlags['start'] or detectionOperationFlags['stop'] or detectionOperationFlags['pause']):
        return jsonify({'error': 'Operation in progress, please wait...'}), 429
    if(statusDict['detectionStatus']):
        return jsonify({'error': 'Detection has not been stopped, please stop it first...'}), 429
    detectionOperationFlags['start'] = True
    socketio.emit('clear', {})
    data = request.json
    print(data)
    # youtube_url = data['youtubeUrl']
    drone_direction = data['droneDirection']
    highwayName = data['highwayName']
    videoDuration = data['videoDuration']
    if(videoDuration=="1 minute"):
        video_duration = 60
    elif (videoDuration=="2 minutes"):
        video_duration = 120
    else:
        video_duration = 120 # default value
    if drone_direction in ("From left to right", "ToRight"):
        video_direction = "left"
    elif drone_direction in ("From right to left", "ToLeft"):
        video_direction = "right"
    else:
        video_direction = "Omission" # default value
    # Read current datetime and create corresponding folders
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experimentFolder = rootFolder + highwayName + '/' + timestamp + '/'
    os.makedirs(experimentFolder, exist_ok=True)
    # Make sure each folder exists
    for eachFolder in subFolders:
        if(not os.path.exists(experimentFolder+eachFolder)):
            os.makedirs(experimentFolder+eachFolder, exist_ok=True)
    # Update outputFolder
    for eachName in outputFilesName:
        outputFiles[eachName] = experimentFolder + 'recordsFolder/' + eachName + '.txt'
        with open(outputFiles[eachName], 'w') as file:
            file.write('')
    detectionResultsFile = experimentFolder + 'detectionResults.txt'
    with open(detectionResultsFile, 'w') as file:
        file.write('')
    for key, value in shutdownEvents.items():
        value.clear()
    # processHandler['stream'] = Process(target=streamCaptureYouTube, args=(youtube_url, experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
    # processHandler['stream'] = Process(target=streamCaptureRTMP, args=(rtmp_url, experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
    # processHandler['stream'] = Process(target=streamCaptureTwitch, args=(experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
    if local_video_path:
        processHandler['stream'] = Process(target=streamCaptureLocal, args=(local_video_path, experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
    elif camera_source is not None:
        processHandler['stream'] = Process(target=streamCaptureCamera, args=(camera_source, experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
    else:
        processHandler['stream'] = Process(target=streamCaptureTwitch, args=(experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
    processHandler['video1'] = Process(target = autoProcessing, args = (experimentFolder, experimentFolder+'watchFolder1/', shutdownEvents['video'], outputFiles['videoProcessingLogs1'], outputFiles['detectionResultsLogs'], dashboard_url))
    processHandler['video2'] = Process(target = autoProcessing, args = (experimentFolder, experimentFolder+'watchFolder2/', shutdownEvents['video'], outputFiles['videoProcessingLogs2'], outputFiles['detectionResultsLogs'], dashboard_url))
    processHandler['video3'] = Process(target = autoProcessing, args = (experimentFolder, experimentFolder+'watchFolder3/', shutdownEvents['video'], outputFiles['videoProcessingLogs3'], outputFiles['detectionResultsLogs'], dashboard_url))
    processHandler['stream'].start()
    statusDict['streamCaptureStatus'] = True
    # Start realtime processor for live MJPEG feed on detection page
    global realtime_proc
    if camera_source is not None and not local_video_path:
        if not realtime_proc or not realtime_proc._running:
            models_dir = os.path.join(_project_dir, 'models')
            rt_output_dir = os.path.join(rootFolder, 'realtime')
            os.makedirs(rt_output_dir, exist_ok=True)
            realtime_proc = RealtimeProcessor(
                camera_source=camera_source,
                models_dir=models_dir,
                output_dir=rt_output_dir,
                video_direction=video_direction,
                dashboard_url=dashboard_url,
            )
            realtime_proc.start()
    statusDict['gpsFetchStatus'] = True
    # socketio.emit('mainOutput', {'message': 'Stream capture process started.'}, room='logs')
    with open(outputFiles['mainOutput'], 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('Stream capture process started!\n')
        portalocker.unlock(file)
    processHandler['video1'].start()
    processHandler['video2'].start()
    processHandler['video3'].start()
    # processHandler['video4'].start()
    statusDict['videoProcessingStatus'] = True
    # socketio.emit('mainOutput', {'message': 'Video processing process started.'}, room='logs')
    with open(outputFiles['mainOutput'], 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('Video processing process started.\n')
        portalocker.unlock(file)
    # socketio.start_background_task(emitVideoProcessingResults, videoProcessingParentPipe, threadShutdwonEvent)
    socketio.start_background_task(emitOutputFiles, outputFiles, shutdownEvents['thread'])
    statusDict['detectionStatus'] = True
    # statusDict['youtubeUrl'] = youtube_url
    statusDict['highwayName'] = highwayName
    statusDict['droneDirection'] = drone_direction
    statusDict['videoDuration'] = videoDuration
    statusDict['pauseStatus'] = False
    socketio.emit('updateStatus', statusDict, room="detection")
    detectionOperationFlags['start'] = False
    # Process the data here or call other Python functions with these values
    # return jsonify({'status': 'success', 'youtubeUrl': youtube_url, 'droneDirection': drone_direction}), 200
    return jsonify({'status': 'success', 'droneDirection': drone_direction}), 200


@app.route('/stop-detection', methods=['POST'])
@login_required
def stop_detection():
    global statusDict, detectionOperationFlags
    if (detectionOperationFlags['start'] or detectionOperationFlags['stop'] or detectionOperationFlags['pause']):
        return jsonify({'error': 'Operation in progress, please wait...'}), 429
    if(not statusDict['detectionStatus']):
        return jsonify({'error': 'Detection has not been strated, please start it first...'}), 429
    detectionOperationFlags['stop'] = True
    shutdownEvents['stream'].set()
    shutdownEvents['video'].set()
    # Stop realtime processor if running (non-blocking)
    global realtime_proc
    if realtime_proc and realtime_proc._running:
        realtime_proc._running = False
    # Force-terminate all child processes
    with open(outputFiles['mainOutput'], 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('Stopping all processes...\n')
        portalocker.unlock(file)
    for pname in ['stream', 'video1', 'video2', 'video3']:
        proc = processHandler.get(pname)
        if proc is not None:
            try:
                proc.terminate()
                proc.join(timeout=3)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=2)
            except Exception:
                pass
    statusDict['videoProcessingStatus'] = False
    statusDict['streamCaptureStatus'] = False
    statusDict['gpsFetchStatus'] = False
    statusDict['detectionStatus'] = False
    statusDict['pauseStatus'] = False
    shutdownEvents['thread'].set()
    socketio.emit('updateStatus', statusDict, room='detection')
    with open(outputFiles['mainOutput'], 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('All processes stopped.\n')
        portalocker.unlock(file)
    detectionOperationFlags['stop'] = False
    return jsonify({'status': 'success'}), 200

def toggle_state():
    # global processHandler, detectionOperationFlags, statusDict, youtube_url, drone_direction, highwayName, video_duration
    global processHandler, detectionOperationFlags, statusDict, drone_direction, highwayName, video_duration
    if (detectionOperationFlags['start'] or detectionOperationFlags['stop'] or detectionOperationFlags['pause']):
        return jsonify({'error': 'Operation in progress, please wait...'}), 429
    if(not statusDict['detectionStatus']):
        return jsonify({'error': 'Detection has not been strated, please start it first...'}), 429
    detectionOperationFlags['pause'] = True
    # current_state = request.json.get('current_state')
    if statusDict['pauseStatus']:
        # 在这里添加代码来恢复你的程序
        # new_state = 'Pause Detection'
        data = request.json
        print(data)
        # youtube_url = data['youtubeUrl']
        drone_direction = data['droneDirection']
        # highwayName = data['highwayName']
        videoDuration = data['videoDuration']
        if(videoDuration=="1 minute"):
            video_duration = 60
        elif (videoDuration=="2 minutes"):
            video_duration = 120
        else:
            video_duration = 120 # default value
        if drone_direction in ("From left to right", "ToRight"):
            video_direction = "left"
        elif drone_direction in ("From right to left", "ToLeft"):
            video_direction = "right"
        else:
            video_direction = "Omission" # default value
        statusDict['droneDirection'] = drone_direction
        statusDict['videoDuration'] = videoDuration
        # Start the stream capture process
        shutdownEvents['stream'].clear()
        if local_video_path:
            processHandler['stream'] = Process(target=streamCaptureLocal, args=(local_video_path, experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
        elif camera_source is not None:
            processHandler['stream'] = Process(target=streamCaptureCamera, args=(camera_source, experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
        else:
            processHandler['stream'] = Process(target=streamCaptureTwitch, args=(experimentFolder, video_direction, shutdownEvents['stream'], outputFiles['streamCaptureLogs'], video_duration))
        processHandler['stream'].start()
        # socketio.emit('mainOutput', {'message': 'Stream Capture & GPS fetch process started from pausing.'}, room='logs')
        with open(outputFiles['mainOutput'], 'a') as file:
            portalocker.lock(file, portalocker.LOCK_EX)
            file.write('Stream Capture & GPS fetch process started from pausing.\n')
            portalocker.unlock(file)
        statusDict['streamCaptureStatus'] = True
        # statusDict['gpsFetchStatus'] = True
        statusDict['pauseStatus'] = False
        socketio.emit('updateStatus', statusDict, room='detection')
    else:
        # 在这里添加代码来暂停你的程序
        # new_state = 'Resume Detection'
        shutdownEvents['stream'].set()
        while (statusDict['streamCaptureStatus']):
            time.sleep(1)
            if(not processHandler['stream'].is_alive()):
                # socketio.emit('mainOutput', {'message': 'Stream Capture & GPS fetch process paused.'}, room='logs')
                with open(outputFiles['mainOutput'], 'a') as file:
                    portalocker.lock(file, portalocker.LOCK_EX)
                    file.write('Stream Capture & GPS fetch process paused.\n')
                    portalocker.unlock(file)
                statusDict['streamCaptureStatus'] = False
                # statusDict['gpsFetchStatus'] = False
                statusDict['pauseStatus'] = True
                socketio.emit('updateStatus', statusDict, room='detection')
    detectionOperationFlags['pause'] = False
    # return jsonify(new_state=new_state), 200
    return jsonify({'status': 'success'}), 200

@app.route('/restart-server', methods=['POST'])
@login_required
def restart():
    print("Restarting the Flask app...")
    for key, value in processHandler.items():
        if(value is not None):
            value.terminate()
            value.join()
    os._exit(0)  # 让Flask应用退出

@app.route('/get-highway-list', methods=['GET'])
@login_required
def get_list_a():
    # 返回列表A的数据，示例为静态列表
    highwayFolders = [d for d in os.listdir(rootFolder) if os.path.isdir(os.path.join(rootFolder, d))]
    return jsonify(highwayFolders)

@app.route('/get-datetime-list', methods=['POST'])
@login_required
def get_list_b():
    data = request.get_json()
    selectedItem = data['selectedItem']
    highwayFolder = rootFolder + selectedItem + '/'
    datetimeFolders = [d for d in os.listdir(highwayFolder) if os.path.isdir(os.path.join(highwayFolder, d))]
    # 根据selectedItem查询并返回列表B的数据，示例为静态列表
    return jsonify(datetimeFolders)

@app.route('/get-detection-result', methods=['POST'])
@login_required
def get_final_result():
    data = request.get_json()
    fileContent = None
    selectedItem = data['selectedItem']
    # print(selectedItem)
    resultsFile = rootFolder + selectedItem + 'detectionResults.txt'
    imageFolder = os.path.dirname(resultsFile)  # 获取 detectionResults.txt 所在文件夹
    if(os.path.exists(resultsFile)):
        # read detectionResultsFile
        with open(resultsFile, 'r') as file:
            # print(f"{key} reading!\n")
            portalocker.lock(file, portalocker.LOCK_SH)
            fileContent = file.read()
            portalocker.unlock(file)
        # # 查找该文件夹下的 .png 图片
        # png_files = [f for f in os.listdir(imageFolder) if f.endswith('.png')]
        # Search for congestion.txt file and get its content
        congestion_path = os.path.join(imageFolder, 'congestion.txt')
        congestion_content = ''
        if os.path.exists(congestion_path):
            with open(congestion_path, 'r') as f:
                congestion_content = f.read()
        # return jsonify({'results': fileContent, 'images': png_files, 'congestion': congestion_content}), 200
        return jsonify({'results': fileContent, 'congestion': congestion_content}), 200
    else:
        return jsonify({'error': 'No detection results file found under this option'}), 404
    # 根据selectedItem查询并返回最终结果，示例为静态文本

@app.route('/video/<path:filename>')  # 使用 path: 来匹配包含斜杠的路径
@login_required
def video(filename):
    # 定义基础目录，确保路径规范化
    base_directory = os.path.normpath(rootFolder)
    # 安全地构建完整的文件路径
    filepath = os.path.normpath(os.path.join(base_directory, filename))
    # 使用规范化后的路径进行安全检查
    if os.path.commonpath([base_directory, filepath]) != base_directory:
        abort(403)  # 如果文件路径超出了基础目录，则返回403禁止访问错误
    # 检查文件是否真实存在
    # print(filepath)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return send_from_directory(base_directory, filename)
    else:
        abort(404)  # 文件不存在时返回404错误

@app.route('/get-current-detection-result')
@login_required
def get_current_detection_result():
    # Always check the latest live_camera folder for results
    best = detectionResultsFile
    live_dir = os.path.join(rootFolder, 'live_camera')
    if os.path.isdir(live_dir):
        subs = sorted([d for d in os.listdir(live_dir)
                       if os.path.isdir(os.path.join(live_dir, d))], reverse=True)
        for sub in subs:
            candidate = os.path.join(live_dir, sub, 'detectionResults.txt')
            if os.path.exists(candidate) and os.path.getsize(candidate) > 0:
                best = candidate
                break
    if best and os.path.exists(best) and os.path.getsize(best) > 0:
        with open(best, 'r') as file:
            portalocker.lock(file, portalocker.LOCK_SH)
            content = file.read().strip()
            portalocker.unlock(file)
        if content:
            return jsonify({'results': content}), 200
    return jsonify({'results': ''}), 200


@app.route('/get-live-traffic-status')
@login_required
def get_live_traffic_status():
    """Return real-time traffic classification status for dashboard polling."""
    if not experimentFolder:
        return jsonify({'status': 'idle', 'label': 'Waiting for detection...', 'class_id': -1}), 200
    status_file = os.path.join(experimentFolder, 'live_status.json')
    if os.path.exists(status_file):
        try:
            with open(status_file, 'r') as f:
                import portalocker
                portalocker.lock(f, portalocker.LOCK_SH)
                data = json.load(f)
                portalocker.unlock(f)
            return jsonify(data), 200
        except Exception:
            pass
    return jsonify({'status': 'idle', 'label': 'Processing...', 'class_id': -1}), 200

@app.route('/source-video')
@login_required
def source_video():
    if camera_source is not None and not local_video_path:
        # Live mode: redirect to MJPEG stream
        return redirect('/realtime-feed')
    if local_video_path and os.path.exists(local_video_path):
        return send_from_directory(os.path.dirname(local_video_path), os.path.basename(local_video_path))
    abort(404)

@app.route('/generate-vlm-report', methods=['POST'])
@login_required
def generate_vlm_report():
    global vlm_report_status
    if vlm_report_status['status'] == 'running':
        return jsonify({'status': 'already_running'}), 429
    if not experimentFolder:
        return jsonify({'error': 'No experiment running'}), 400

    # Find the latest vlm_evidence folder
    evidence_base = os.path.join(experimentFolder, 'vlm_evidence')
    if not os.path.exists(evidence_base):
        return jsonify({'error': 'No incident evidence found. Run detection first.'}), 404

    evidence_dirs = [d for d in os.listdir(evidence_base)
                     if os.path.isdir(os.path.join(evidence_base, d))]
    if not evidence_dirs:
        return jsonify({'error': 'No incident evidence folders found'}), 404

    # Use the most recent evidence directory
    latest_dir = os.path.join(evidence_base, sorted(evidence_dirs)[-1])
    manifest_path = os.path.join(latest_dir, 'manifest.json')
    if not os.path.exists(manifest_path):
        return jsonify({'error': 'No manifest found in evidence folder'}), 404

    vlm_report_status = {'status': 'running', 'results': None, 'error': None}

    def run_vlm_analysis():
        global vlm_report_status
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            image_paths = []
            for fname in manifest.get('evidence_frames', []):
                p = os.path.join(latest_dir, fname)
                if os.path.exists(p):
                    image_paths.append(p)
            for fname in manifest.get('trajectory_images', []):
                p = os.path.join(latest_dir, fname)
                if os.path.exists(p):
                    image_paths.append(p)

            if not image_paths:
                vlm_report_status = {'status': 'error', 'results': None, 'error': 'No evidence images found'}
                return

            results = analyze_incident(image_paths)
            video_name = manifest.get('video_name', 'unknown')
            vlm_output_dir = os.path.join(experimentFolder, 'vlm_reports')
            save_vlm_results(results, vlm_output_dir, video_name)

            vlm_report_status = {
                'status': 'complete',
                'results': results,
                'video_name': video_name,
                'images_analyzed': len(image_paths),
                'error': None,
            }
        except Exception as e:
            vlm_report_status = {'status': 'error', 'results': None, 'error': str(e)}

    thread = threading.Thread(target=run_vlm_analysis, daemon=True)
    thread.start()
    return jsonify({'status': 'started'}), 200


@app.route('/get-vlm-report')
@login_required
def get_vlm_report():
    return jsonify(vlm_report_status), 200


@app.route('/vlm-evidence-image/<path:filename>')
@login_required
def vlm_evidence_image(filename):
    if not experimentFolder:
        abort(404)
    evidence_base = os.path.join(experimentFolder, 'vlm_evidence')
    filepath = os.path.normpath(os.path.join(evidence_base, filename))
    if not filepath.startswith(os.path.normpath(evidence_base)):
        abort(403)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return send_from_directory(os.path.dirname(filepath), os.path.basename(filepath))
    abort(404)


@app.route('/img/<path:filename>')  # 使用 path: 来匹配包含斜杠的路径
@login_required
def img(filename):
    # 定义基础目录，确保路径规范化
    base_directory = os.path.normpath(rootFolder)
    # 安全地构建完整的文件路径
    filepath = os.path.normpath(os.path.join(base_directory, filename))
    # 使用规范化后的路径进行安全检查
    if os.path.commonpath([base_directory, filepath]) != base_directory:
        abort(403)  # 如果文件路径超出了基础目录，则返回403禁止访问错误
    # 检查文件是否真实存在
    # print(filepath)
    if os.path.exists(filepath) and os.path.isfile(filepath):
        return send_from_directory(base_directory, filename)
    else:
        abort(404)  # 文件不存在时返回404错误

# Customized funtions
# # show detection results based on Pipes.
# def emitVideoProcessingResults(parentPipe, threadShutdwonEvent):
#     while not threadShutdwonEvent.is_set():  # 检查事件是否被设置
#         socketio.emit('emitVideoProcessingResults', {'message': "Checking results..."}, room='logs')
#         try:
#             if(parentPipe.poll()):
#                 msg = parentPipe.recv()
#                 socketio.emit('detectionResults', {'message': msg}, room='detection')
#                 socketio.emit('emitVideoProcessingResults', {'message': f"Receiving result: {msg}"}, room='logs')
#         except Exception as err:
#             """ 当out_pipe接受不到输出的时候且输入被关闭的时候，会抛出EORFError，可以捕获并且退出子进程 """
#             print(f"Auto detection process is stoped because of {err}\n")
#             socketio.emit('emitVideoProcessingResults', {'message': f"Auto detection process is stoped because of {err}"})
#             if((parentPipe is not None) and (not parentPipe.closed)):
#                 parentPipe.close()
#                 parentPipe = None
#             break
#         time.sleep(3)
#     socketio.emit('mainOutput', {'message': 'emitVideoProcessingResults thread stopped!'}, room='logs')

def emitOutputFiles(outputFiles, threadShutdwonEvent):
    time.sleep(5)   # Wait to let every process starts first
    while not threadShutdwonEvent.is_set():  # 检查事件是否被设置
        socketio.emit('emitOutputFiles', {'message': f"Checking files...{datetime.now().strftime('%Y%m%d_%H%M%S')}"}, room='logs')
        fileContent = None
        for key, value in outputFiles.items():
            if(os.path.exists(value)):
                with open(value, 'r') as file:
                    portalocker.lock(file, portalocker.LOCK_SH)
                    fileContent = file.read()
                    portalocker.unlock(file)
                socketio.emit(key, {'message': fileContent}, room='logs')
                if(key=='detectionResultsLogs'):
                    print('Enter detectionResultsLogs part!')
                    detectionResultsFileContent = None
                    with open(detectionResultsFile, 'r') as file:
                        portalocker.lock(file, portalocker.LOCK_SH)
                        detectionResultsFileContent = file.read()
                        portalocker.unlock(file)
                    # For play local videos
                    # socketio.emit('detectionResults', {'message': detectionResultsFileContent}, room='detection')
                    detectionResultsFileLines = detectionResultsFileContent.split('\n')
                    # Begin to process and combine the GPS coordinates
                    fileLines = fileContent.split('\n')
                    detectionResultsFileLines = [item for item in detectionResultsFileLines if item != ""]
                    fileLines = [item for item in fileLines if item != ""]
                    # print(f'detectionResultsFileLines: {detectionResultsFileLines}, fileLines: {fileLines}')
                    if((len(detectionResultsFileLines) < len(fileLines))):
                        # 从B中删除所有包含A中所有字符串的元素
                        if not detectionResultsFileLines:
                            filteredFileLines = fileLines[:]
                        else:
                            # 提取detectionResultsFileLines中每项的基础部分（前三个组成部分）
                            base_detection_results = [','.join(line.split(',')[:3]) for line in detectionResultsFileLines]
                            filteredFileLines = [item for item in fileLines if item not in base_detection_results]
                        print(f'filteredFileLines: {filteredFileLines}')

                        if os.path.exists(outputFiles['gpsFetchResults']) and os.path.getsize(outputFiles['gpsFetchResults']) > 0:
                            # GPS data available — correlate with GPS coordinates
                            gpsDf = pd.read_csv(outputFiles['gpsFetchResults'], header=None, names=['datetime', 'latitude', 'longitude'])
                            gpsDf['datetime'] = pd.to_datetime(gpsDf['datetime'], format='%Y%m%d_%H%M%S')
                            gpsDf['latitude'] = gpsDf['latitude'].astype('float64')
                            gpsDf['longitude'] = gpsDf['longitude'].astype('float64')
                            for i in range(0, len(filteredFileLines)):
                                datetimeIdx = datetime.strptime(filteredFileLines[i].split(',')[0].split('-')[0], "%Y%m%d_%H%M%S")
                                start_time = datetimeIdx
                                end_time = datetimeIdx + timedelta(seconds=math.ceil(float(filteredFileLines[i].split(',')[1])))
                                filtered_df = gpsDf[(gpsDf['datetime'] >= start_time) & (gpsDf['datetime'] <= end_time)]
                                latitude_mean = filtered_df['latitude'].mean()
                                longitude_mean = filtered_df['longitude'].mean()
                                print(f'latitude_mean: {latitude_mean}, longitude_mean: {longitude_mean}')
                                if ((not pd.isna(latitude_mean)) and (not pd.isna(longitude_mean))):
                                    resStr = filteredFileLines[i] + ',' + str(latitude_mean) + ',' + str(longitude_mean)
                                    with open(detectionResultsFile, 'a') as file:
                                        portalocker.lock(file, portalocker.LOCK_EX)
                                        file.write(resStr + '\n')
                                        portalocker.unlock(file)
                        else:
                            # No GPS data (local testing) — use default coordinates
                            for i in range(0, len(filteredFileLines)):
                                resStr = filteredFileLines[i] + ',28.0587,-82.4139'
                                with open(detectionResultsFile, 'a') as file:
                                    portalocker.lock(file, portalocker.LOCK_EX)
                                    file.write(resStr + '\n')
                                    portalocker.unlock(file)

                        # Read file content of detectionResultsFile and emit
                        with open(detectionResultsFile, 'r') as file:
                            portalocker.lock(file, portalocker.LOCK_SH)
                            fileContent = file.read()
                            portalocker.unlock(file)
                        socketio.emit('detectionResults', {'message': fileContent}, room='detection')
                    
        time.sleep(3)
    # socketio.emit('mainOutput', {'message': 'emitOutputFiles thread stopped!'}, room='logs')
    with open(outputFiles['mainOutput'], 'a') as file:
        portalocker.lock(file, portalocker.LOCK_EX)
        file.write('emitOutputFiles thread stopped!\n')
        portalocker.unlock(file)


# ── Real-Time Detection Routes ──────────────────────────────────────────

@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtime.html', active_page='realtime')


@app.route('/start-realtime', methods=['POST'])
@login_required
def start_realtime():
    global realtime_proc
    if realtime_proc and realtime_proc._running:
        return jsonify({'error': 'Real-time detection already running'}), 429

    data = request.json or {}
    source = data.get('source', 0)  # default: /dev/video0 (HDMI capture)
    direction = data.get('direction', 'left')

    # Accept integer sources
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    global experimentFolder, detectionResultsFile
    models_dir = os.path.join(_project_dir, 'models')
    output_dir = os.path.join(rootFolder, 'realtime')
    os.makedirs(output_dir, exist_ok=True)

    # Create experiment folder so results show on dashboard
    from datetime import datetime as _dt
    timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
    experimentFolder = os.path.join(rootFolder, 'live_camera', timestamp) + '/'
    os.makedirs(experimentFolder, exist_ok=True)
    for sf in ['recordsFolder', 'watchFolder1', 'trajPlots']:
        os.makedirs(os.path.join(experimentFolder, sf), exist_ok=True)
    detectionResultsFile = os.path.join(experimentFolder, 'detectionResults.txt')
    open(detectionResultsFile, 'w').close()

    realtime_proc = RealtimeProcessor(
        camera_source=source,
        models_dir=models_dir,
        output_dir=output_dir,
        video_direction=direction,
        dashboard_url=dashboard_url,
        detection_results_file=detectionResultsFile,
        experiment_folder=experimentFolder,
    )
    realtime_proc.start()
    return jsonify({'status': 'started', 'source': str(source), 'direction': direction})


@app.route('/stop-realtime', methods=['POST'])
@login_required
def stop_realtime():
    global realtime_proc
    if realtime_proc:
        realtime_proc.stop()
    return jsonify({'status': 'stopped'})


@app.route('/realtime-status')
@login_required
def realtime_status():
    gate_status = latest_outri_gate_status()
    if realtime_proc:
        status = dict(realtime_proc.status)
        status["outri_gate"] = gate_status
        return jsonify(status)
    return jsonify({'running': False, 'status': 'idle', 'outri_gate': gate_status})


@app.route('/realtime-feed')
@login_required
def realtime_feed():
    """MJPEG stream of annotated frames from real-time detector."""
    def generate():
        while True:
            if realtime_proc:
                frame = realtime_proc.annotated_frame
                if frame is not None:
                    _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, 50])
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')



def _auto_start_realtime():
    """Auto-start realtime detection pipeline after a short delay."""
    import time as _t
    import threading

    def _delayed_start():
        _t.sleep(8)  # wait for Flask + GPU to be ready
        try:
            global realtime_proc, experimentFolder, detectionResultsFile
            if realtime_proc and realtime_proc._running:
                print("Realtime already running, skipping auto-start", flush=True)
                return

            from datetime import datetime as _dt
            models_dir = os.path.join(_project_dir, 'models')
            output_dir = os.path.join(rootFolder, 'realtime')
            os.makedirs(output_dir, exist_ok=True)

            timestamp = _dt.now().strftime("%Y%m%d_%H%M%S")
            experimentFolder = os.path.join(rootFolder, 'live_camera', timestamp) + '/'
            os.makedirs(experimentFolder, exist_ok=True)
            for sf in ['recordsFolder', 'watchFolder1', 'trajPlots']:
                os.makedirs(os.path.join(experimentFolder, sf), exist_ok=True)
            detectionResultsFile = os.path.join(experimentFolder, 'detectionResults.txt')
            open(detectionResultsFile, 'w').close()

            realtime_proc = RealtimeProcessor(
                camera_source=0,
                models_dir=models_dir,
                output_dir=output_dir,
                video_direction='left',
                dashboard_url=dashboard_url,
                detection_results_file=detectionResultsFile,
                experiment_folder=experimentFolder,
            )
            realtime_proc.start()
            print("Auto-started realtime pipeline on camera 0", flush=True)
        except Exception as e:
            print(f"Auto-start failed: {e}", flush=True)

    t = threading.Thread(target=_delayed_start, daemon=True)
    t.start()

if __name__ == '__main__':
    _auto_start_realtime()
    socketio.run(app, debug=False, host='0.0.0.0', port=4430, allow_unsafe_werkzeug=True)
    # socketio.run(app, debug=False, host='0.0.0.0', ssl_context=("C:/Users/baili/Documents/MEGA_Sync/IncidentDetectionProject/detection.duckdns.org/combined.crt", "C:/Users/baili/Documents/MEGA_Sync/IncidentDetectionProject/detection.duckdns.org/private.key"), port=443)


