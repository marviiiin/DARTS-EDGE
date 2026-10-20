"""Edge-to-Dashboard communication module.

After the Jetson detects an incident, this module sends the detection results
and evidence frames to the remote dashboard server, which then triggers
VLM analysis and displays the report.
"""
import os
import json
import base64
import time
import requests
import cv2 as cv


class EdgeCommunicator:
    def __init__(self, dashboard_url, auth_user="admin", auth_password="usfsumlab"):
        """
        Args:
            dashboard_url: Base URL of the dashboard (e.g., "http://192.168.1.100:4430")
            auth_user: Dashboard login username
            auth_password: Dashboard login password
        """
        self.dashboard_url = dashboard_url.rstrip('/')
        self.auth_user = auth_user
        self.auth_password = auth_password
        self.session = requests.Session()
        self._logged_in = False

    def _login(self):
        """Authenticate with the dashboard."""
        try:
            # Get the login page first (for CSRF if needed)
            self.session.get(f'{self.dashboard_url}/login', timeout=10)
            # POST login credentials
            resp = self.session.post(
                f'{self.dashboard_url}/login',
                data={'username': self.auth_user, 'password': self.auth_password},
                allow_redirects=False,
                timeout=10,
            )
            self._logged_in = resp.status_code in (200, 302)
            return self._logged_in
        except Exception as e:
            print(f'[EdgeComm] Login failed: {e}')
            return False

    def send_detection_result(self, video_name, duration, detection_label,
                              evidence_dir=None, manifest=None):
        """Send detection result and evidence images to the dashboard.

        Args:
            video_name: Name of the processed video
            duration: Video duration in seconds
            detection_label: "Incident Detected!", "Congestion Detected.", or "No Incident."
            evidence_dir: Path to vlm_evidence folder with frames and trajectories
            manifest: Dict with detection metadata (from manifest.json)

        Returns:
            True if successfully sent, False otherwise
        """
        if not self._logged_in:
            if not self._login():
                print('[EdgeComm] Cannot authenticate with dashboard')
                return False

        # Build the payload
        payload = {
            'video_name': video_name,
            'duration': duration,
            'detection_label': detection_label,
            'source': 'edge_device',
            'timestamp': time.strftime('%Y%m%d_%H%M%S'),
        }

        if manifest:
            payload['manifest'] = manifest

        # Encode evidence images as base64
        evidence_images = {}
        if evidence_dir and os.path.isdir(evidence_dir):
            for fname in os.listdir(evidence_dir):
                if fname.endswith('.png') or fname.endswith('.jpg'):
                    fpath = os.path.join(evidence_dir, fname)
                    with open(fpath, 'rb') as f:
                        evidence_images[fname] = base64.b64encode(f.read()).decode('utf-8')

        payload['evidence_images'] = evidence_images

        try:
            resp = self.session.post(
                f'{self.dashboard_url}/edge-detection-report',
                json=payload,
                timeout=30,
            )
            if resp.status_code == 200:
                result = resp.json()
                print(f'[EdgeComm] Result sent successfully: {result.get("status")}')
                return True
            else:
                print(f'[EdgeComm] Send failed: HTTP {resp.status_code}')
                # Re-login on auth failure
                if resp.status_code in (401, 403):
                    self._logged_in = False
                return False
        except Exception as e:
            print(f'[EdgeComm] Send error: {e}')
            self._logged_in = False
            return False

    def send_frame(self, frame, max_width=640, jpeg_quality=60):
        """Send a video frame to the dashboard for live streaming.

        Args:
            frame: OpenCV BGR image (numpy array)
            max_width: Resize frame to this width to save bandwidth
            jpeg_quality: JPEG compression quality (0-100)

        Returns:
            True if successfully sent, False otherwise
        """
        if not self._logged_in:
            if not self._login():
                return False

        try:
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv.resize(frame, (max_width, int(h * scale)))

            _, buf = cv.imencode('.jpg', frame, [cv.IMWRITE_JPEG_QUALITY, jpeg_quality])
            frame_b64 = base64.b64encode(buf).decode('utf-8')

            resp = self.session.post(
                f'{self.dashboard_url}/edge-stream-frame',
                json={'frame': frame_b64, 'timestamp': time.time()},
                timeout=5,
            )
            return resp.status_code == 200
        except Exception:
            return False

    def send_heartbeat(self, status_info=None):
        """Send a heartbeat to let the dashboard know the edge device is alive."""
        if not self._logged_in:
            if not self._login():
                return False
        try:
            payload = {
                'source': 'edge_device',
                'timestamp': time.strftime('%Y%m%d_%H%M%S'),
                'status': status_info or 'running',
            }
            resp = self.session.post(
                f'{self.dashboard_url}/edge-heartbeat',
                json=payload,
                timeout=10,
            )
            return resp.status_code == 200
        except Exception:
            self._logged_in = False
            return False
