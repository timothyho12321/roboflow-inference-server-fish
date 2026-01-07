import cv2
import os
import yaml
from dotenv import load_dotenv
import time
import math
import queue
import threading
import numpy as np
import supervision as sv
from inference_sdk import InferenceHTTPClient
# import aravis # Ensure the aravis python bindings are installed (gi.repository)
from gi.repository import Aravis

# --- 1. USER CONFIGURATION ---
MAX_ALLOWED_TILT = 50.0 
CONFIDENCE_THRESHOLD = 0.45 
IOU_THRESHOLD = 0.40

# ROBOFLOW SERVER SETTINGS
load_dotenv()
API_URL = "http://localhost:9001"
API_KEY = os.getenv("API_KEY")


# Keypoint Definitions (Tiger Barbs)
KEYPOINT_NAMES = ["S", "D", "T", "C", "B"]
IDX_DORSAL = 1  
IDX_BELLY  = 4  

# --- 2. CAMERA CLASS (Your Aravis Driver) ---
class AravisCaptureThread:
    def __init__(self, ip_address, name="Cam"):
        self.ip = ip_address
        self.name = name
        self.stop_event = threading.Event()
        self.image_queue = queue.Queue(maxsize=2) # Keep buffer small for low latency
        self.cam = None
        self.stream = None
        
        # Settings
        self.width = None
        self.height = None
        self.fps_limit = 30.0
        self.desired_pixel_format = "BayerRG8" 
        self.exposure_time = 20000.0
        self.gain = 15.0
        
        # FPS Calc
        self.fps_data = {'start_time': time.time(), 'frame_count': 0, 'fps': 0, 'last_time': time.time()}

    def start(self):
        self.thread = threading.Thread(target=self.run)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if hasattr(self, 'thread'):
            self.thread.join()

    def run(self):
        print(f"[{self.name}] Connecting to Aravis Camera at {self.ip}...")
        try:
            self.cam = Aravis.Camera.new(self.ip)
        except:
            print(f"[{self.name}] Failed to find camera.")
            return

        # Basic Setup
        self.cam.set_string("AcquisitionMode", "Continuous")
        self.cam.set_float("AcquisitionFrameRate", self.fps_limit)
        
        # Auto-detect resolution
        self.width = self.cam.get_integer("Width")
        self.height = self.cam.get_integer("Height")
        payload = self.cam.get_payload()

        # Stream Setup
        self.stream = self.cam.create_stream(None, None)
        for _ in range(10): # Allocate buffers
            self.stream.push_buffer(Aravis.Buffer.new_allocate(payload))

        self.cam.start_acquisition()
        print(f"[{self.name}] Acquisition Started. Resolution: {self.width}x{self.height}")

        while not self.stop_event.is_set():
            buffer = self.stream.timeout_pop_buffer(1000000)
            if buffer:
                if buffer.get_status() == Aravis.BufferStatus.SUCCESS:
                    data = buffer.get_data()
                    # Basic Debayering (Assumes BayerRG8 for simplicity, adjust if needed)
                    frame = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_BAYER_BG2BGR)
                    
                    # Manage Queue (Drop old frames if processing is slow)
                    if not self.image_queue.empty():
                        try: self.image_queue.get_nowait()
                        except: pass
                    self.image_queue.put(frame_bgr)
                
                self.stream.push_buffer(buffer)

        self.cam.stop_acquisition()

# --- 3. HELPER FUNCTION: TILT CALCULATION ---
def get_fish_tilt(kpts_xy):
    dorsal = kpts_xy[IDX_DORSAL]
    belly  = kpts_xy[IDX_BELLY]

    if dorsal[0] == 0 or belly[0] == 0: return None

    dx = dorsal[0] - belly[0]
    dy = dorsal[1] - belly[1] 
    angle_deg = math.degrees(math.atan2(dy, dx))
    
    deviation = abs(angle_deg - (-90))
    if deviation > 180: deviation = 360 - deviation
    return deviation

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    # A. Setup Inference Client
    print("Connecting to Inference Server...")
    client = InferenceHTTPClient(
        api_url=API_URL,
        api_key=API_KEY
    )

    # Load Config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load Model Config
    model_cfg = config.get('model', {})
    # MODEL_ID = model_cfg.get('path_top_view')
    MODEL_ID = model_cfg.get('path_side_view')
    # B. Start Camera (Using ID from config)
    # Using top_source as default. Modify if you want to use side_source or both.
    # camera_id = config.get('top_source') 
    side_camera_id = config.get('side_source')
    
    cam = AravisCaptureThread(side_camera_id)
    cam.start()

    # C. Visualization Annotators
    fish_edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    edge_annotator = sv.EdgeAnnotator(color=sv.Color.YELLOW, thickness=1, edges=fish_edges)
    vertex_annotator = sv.VertexAnnotator(color=sv.Color.GREEN, radius=4)

    print("Starting processing loop. Press 'q' to exit.")
    
    try:
        while True:
            # 1. Get Frame
            if cam.image_queue.empty():
                time.sleep(0.01)
                continue
                
            image = cam.image_queue.get()

            # 2. Inference (Send to Local Docker Container)
            # This replaces 'model(image)'
            result = client.infer(image, model_id=MODEL_ID)
            
            # 3. Convert Results to Supervision Format
            # Note: We use from_inference() instead of from_ultralytics()
            key_points = sv.KeyPoints.from_inference(result)
            
            # Extract detections for bounding boxes
            detections = sv.Detections.from_inference(result)

            # 4. Annotation Logic
            image = edge_annotator.annotate(scene=image, key_points=key_points)
            image = vertex_annotator.annotate(scene=image, key_points=key_points)

            healthy_count = 0
            sick_count = 0

            if len(key_points.xy) > 0:
                for i, kpts in enumerate(key_points.xy):
                    tilt = get_fish_tilt(kpts)
                    
                    # Get BBox coords from Detections
                    x1, y1, x2, y2 = map(int, detections.xyxy[i])

                    if tilt is None:
                        cv2.rectangle(image, (x1, y1), (x2, y2), (128, 128, 128), 2)
                        continue

                    # Classification
                    if tilt <= MAX_ALLOWED_TILT:
                        status, color = "HEALTHY", (0, 255, 0)
                        healthy_count += 1
                    else:
                        status, color = "ABNORMAL", (0, 0, 255)
                        sick_count += 1

                    # Draw Labels
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(image, f"{status} {int(tilt)}deg", (x1, y1 - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 5. Display Stats
            cv2.putText(image, f"Healthy: {healthy_count} | Sick: {sick_count}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Fish Monitor", image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping camera...")
        cam.stop()
        cv2.destroyAllWindows()