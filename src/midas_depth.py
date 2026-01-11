import time
import cv2
import numpy as np
import torch

class MidasDepth:
    def __init__(
        self,
        device="cpu",
        input_size=(384, 216),
        close_threshold=0.65,
        run_every_n_frames=4,
    ):
        self.device = device
        self.w, self.h = input_size
        self.close_threshold = close_threshold
        self.run_every_n_frames = run_every_n_frames
        self.frame_count = 0

        print("Loading MiDaS_small model...")
        self.midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
        self.midas.to(self.device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        self.transform = transforms.small_transform
        print("MiDaS model loaded successfully")

        self.last = {
            "roi_v": 0.0,
            "left_v": 0.0,
            "right_v": 0.0,
            "close_ahead": False,
            "open_dir": "left",
            "turn_dir": "left",
            "timestamp": 0.0,
            "valid": False,
            "depth_vis": None,
        }

    def update(self, frame_bgr):
        self.frame_count += 1
        if self.frame_count % self.run_every_n_frames != 0:
            return self.last

        frame_small = cv2.resize(frame_bgr, (self.w, self.h))
        img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            pred = self.midas(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy()

        dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
        dn = np.clip((depth - dmin) / (dmax - dmin + 1e-6), 0, 1)

        h, w = dn.shape

        center = dn[int(0.50*h):int(0.80*h), int(0.35*w):int(0.65*w)]
        left   = dn[int(0.50*h):int(0.80*h), int(0.05*w):int(0.45*w)]
        right  = dn[int(0.50*h):int(0.80*h), int(0.55*w):int(0.95*w)]

        roi_v = float(np.median(center))
        left_v = float(np.median(left))
        right_v = float(np.median(right))

        close_ahead = roi_v > self.close_threshold
        # Lower normalized values indicate farther/clearer space.
        open_dir = "left" if left_v < right_v else "right"
        turn_dir = "right" if left_v > right_v else "left"

        depth_vis = self._create_visualization(dn)

        self.last = {
            "roi_v": roi_v,
            "left_v": left_v,
            "right_v": right_v,
            "close_ahead": close_ahead,
            "open_dir": open_dir,
            "turn_dir": turn_dir,
            "timestamp": time.time(),
            "valid": True,
            "depth_vis": depth_vis,
        }
        return self.last
    
    def get_calibration_value(self, frame_bgr):
        frame_small = cv2.resize(frame_bgr, (self.w, self.h))
        img = cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB)

        input_batch = self.transform(img).to(self.device)
        with torch.no_grad():
            pred = self.midas(input_batch)
            pred = torch.nn.functional.interpolate(
                pred.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth = pred.cpu().numpy()
        dmin, dmax = np.percentile(depth, 2), np.percentile(depth, 98)
        dn = np.clip((depth - dmin) / (dmax - dmin + 1e-6), 0, 1)

        h, w = dn.shape
        center = dn[int(0.50*h):int(0.80*h), int(0.35*w):int(0.65*w)]
        return float(np.median(center))
    
    def _create_visualization(self, depth_normalized):
        depth_colored = cv2.applyColorMap((depth_normalized * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
        
        h, w = depth_normalized.shape
        
        cv2.rectangle(depth_colored, 
                     (int(0.35*w), int(0.50*h)), 
                     (int(0.65*w), int(0.80*h)), 
                     (0, 0, 255), 2)
        cv2.putText(depth_colored, "CENTER", (int(0.38*w), int(0.47*h)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        cv2.rectangle(depth_colored, 
                     (int(0.05*w), int(0.50*h)), 
                     (int(0.45*w), int(0.80*h)), 
                     (0, 255, 0), 2)
        cv2.putText(depth_colored, "LEFT", (int(0.08*w), int(0.47*h)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.rectangle(depth_colored, 
                     (int(0.55*w), int(0.50*h)), 
                     (int(0.95*w), int(0.80*h)), 
                     (255, 0, 0), 2)
        cv2.putText(depth_colored, "RIGHT", (int(0.58*w), int(0.47*h)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        
        threshold_text = f"Threshold: {self.close_threshold:.2f}"
        cv2.putText(depth_colored, threshold_text, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        depth_vis = cv2.resize(depth_colored, (640, 360))
        
        return depth_vis
