# model_adapters.py

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU for all models
import mediapipe as mp
from ultralytics import YOLO
import time
from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
import torch

class PoseModelAdapter:
    # Base class for pose estimation model adapters
    
    def get_keypoint_mapping(self):
        # Return dict mapping CVAT keypoint names to model's keypoint indices
        raise NotImplementedError
    
    def run_inference(self, images_folder):
        # Run model on all frames. Returns (predictions, inference_time)
        # predictions: dict {frame_num: KeypointResult or None}
        # inference_time: float (seconds)
        raise NotImplementedError
    
    def get_model_name(self):
        # Return model name for saving/plotting
        raise NotImplementedError

class MediaPipeAdapter(PoseModelAdapter):
    # Adapter for MediaPipe Pose Landmarker
    
    def __init__(self, model_name, model_path='../models/mediapipe/pose_landmarker_heavy.task'):
        self.model_path = model_path
        self._model_name = model_name
    
    def get_keypoint_mapping(self):
        return {
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_foot_index': 31,
            'right_foot_index': 32
        }
    
    def run_inference(self, images_folder):
        # Run MediaPipe on all frame
        BaseOptions = mp.tasks.BaseOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=VisionRunningMode.IMAGE
        )
        
        predictions = {}
        start_time = time.time()
        
        with PoseLandmarker.create_from_options(options) as landmarker:
            frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.PNG')])
            
            for frame_file in frame_files:
                frame_num = int(frame_file.split('_')[1].split('.')[0])
                image_path = os.path.join(images_folder, frame_file)
                
                mp_image = mp.Image.create_from_file(image_path)
                result = landmarker.detect(mp_image)
                
                if result.pose_landmarks:
                    predictions[frame_num] = result.pose_landmarks[0]
                else:
                    predictions[frame_num] = None
        
        total_time = time.time() - start_time

        return predictions, total_time
    
    def get_model_name(self):
        if self._model_name:
            return self._model_name
        
        return "MediaPipe"


class YOLOPoseAdapter(PoseModelAdapter):
    """Adapter for YOLO11-pose models (yolo11n-pose, yolo11x-pose, etc.)"""
    
    def __init__(self, model_path, model_name=None):
        """
        Args:
            model_path: Path to .pt file (e.g., 'yolo11n-pose.pt')
            model_name: Optional custom name (otherwise inferred from path)
        """
        self.model_path = model_path
        self._model_name = model_name
    
    def get_keypoint_mapping(self):
        """
        YOLO uses 17 COCO keypoints (indices 0-16).
        We only map the keypoints that overlap with our CVAT annotations.
        
        NOTE: YOLO does NOT have foot_index keypoints, so we exclude them.
        This means YOLO will only be evaluated on 12 keypoints while
        MediaPipe can be evaluated on all 14.
        """
        return {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16,
            # NOTE: No 'left_foot_index' or 'right_foot_index'
        }
    
    def run_inference(self, images_folder):
        """
        Run YOLO inference on all frames.
        
        YOLO returns torch tensors of shape [num_people, 17, 2]
        We need to:
        1. Check if any person detected (dimension 0 > 0)
        2. Pick the best detection if multiple people
        3. Wrap in our KeypointWrapper format
        """
        
        model = YOLO(self.model_path)
        predictions = {}
        
        frame_files = sorted([f for f in os.listdir(images_folder) 
                            if f.endswith('.PNG')])

        start_time = time.time()

        for frame_file in frame_files:
            frame_num = int(frame_file.split('_')[1].split('.')[0])
            image_path = os.path.join(images_folder, frame_file)
            
            # Run inference (verbose=False to suppress output)
            results = model(image_path, verbose=False, device='cpu')
            
            # Check if any person detected
            if len(results) > 0:
                result = results[0]  # YOLO returns list, we use first result
                keypoints_tensor = result.keypoints.xyn  # Shape: [num_people, 17, 2]
                
                # Check if tensor is not empty (at least 1 person detected)
                if keypoints_tensor.shape[0] > 0:
                    # Pick the best person detection
                    if keypoints_tensor.shape[0] == 1:
                        # Only one person - use it
                        person_idx = 0
                    else:
                        # Multiple people - pick highest confidence
                        # (if confidence available, otherwise first person)
                        person_idx = self._select_best_person(result)
                    
                    # Extract keypoints for selected person
                    person_keypoints = keypoints_tensor[person_idx]  # Shape: [17, 2]
                    
                    # Wrap in our format
                    predictions[frame_num] = YOLOKeypointWrapper(person_keypoints)
                else:
                    # Empty tensor - no person detected
                    predictions[frame_num] = None
            else:
                # No results at all
                predictions[frame_num] = None
        
        total_time = time.time() - start_time

        return predictions, total_time
    
    def _select_best_person(self, result):
        """
        Select which person to use when multiple detected.
        Strategy: Use highest confidence score if available, else first person.
        """
        try:
            # Try to get confidence scores (may not always be available)
            if hasattr(result, 'boxes') and result.boxes is not None:
                confidences = result.boxes.conf
                if confidences is not None and len(confidences) > 0:
                    # Return index of highest confidence
                    return int(confidences.argmax())
        except:
            pass
        
        # Fallback: use first person
        return 0
    
    def get_model_name(self):
        """
        Generate name from model path if not provided.
        
        Examples:
            'yolo11n-pose.pt' → 'YOLO11n-Pose'
            'yolo11x-pose.pt' → 'YOLO11x-Pose'
        """
        if self._model_name:
            return self._model_name
        
        filename = os.path.basename(self.model_path)  # 'yolo11n-pose.pt'
        name_part = filename.replace('.pt', '')        # 'yolo11n-pose'
        name_part = name_part.replace('-', ' ')        # 'yolo11n pose'
        
        # Capitalize each part
        parts = name_part.split()
        formatted = ''.join(p.capitalize() for p in parts)  # 'Yolo11nPose'
        
        return formatted


class YOLOKeypointWrapper:
    """
    Wraps YOLO keypoint tensor to match MediaPipe interface.
    
    YOLO gives us a tensor of shape [17, 2] (17 keypoints, x and y)
    We need to make it indexable and have .x and .y attributes like MediaPipe.
    """
    
    def __init__(self, yolo_keypoints):
        """
        Args:
            yolo_keypoints: torch.Tensor of shape [17, 2]
        """
        # Convert torch tensor to numpy for easier handling
        if hasattr(yolo_keypoints, 'cpu'):
            # It's a torch tensor
            self.keypoints = yolo_keypoints.cpu().numpy()
        else:
            # Already numpy
            self.keypoints = yolo_keypoints
    
    def __getitem__(self, idx):
        """
        Make it indexable like MediaPipe landmarks.
        
        Example:
            kp = wrapper[5]  # Get left shoulder
            kp.x  # Access x coordinate
            kp.y  # Access y coordinate
        """
        kp = self.keypoints[idx]
        return KeypointWrapper(x=kp[0], y=kp[1])


class KeypointWrapper:
    """
    Simple wrapper to provide .x and .y attributes.
    Makes YOLO output look like MediaPipe output.
    """
    def __init__(self, x, y, confidence=1.0):
        self.x = float(x)
        self.y = float(y)
        self.confidence = float(confidence)  # 0.0 = missing, >0.0 = detected


class ViTPoseAdapter(PoseModelAdapter):
    """
    Adapter for ViTPose models from HuggingFace transformers.
    
    ViTPose is a top-down approach that requires person detection first.
    We use RT-DETR for person detection, then ViTPose for keypoint estimation.
    
    Supports both ViTPose (simple) and ViTPose++ (plus) models.
    For ViTPose++ models, use dataset_index to select the expert head:
        0: COCO validation
        1: AiC
        2: MPII
        3: AP-10K
        4: APT-36K
        5: COCO-WholeBody
    """
    
    def __init__(self, vitpose_model_name, model_name=None, 
                 person_detector_name="PekingU/rtdetr_r50vd_coco_o365",
                 dataset_index=None):
        """
        Args:
            vitpose_model_name: HuggingFace model name (e.g., 'usyd-community/vitpose-base-simple')
            model_name: Optional custom name for results (otherwise inferred from vitpose_model_name)
            person_detector_name: HuggingFace model name for person detection
            dataset_index: For ViTPose++ only. Integer 0-5 to select expert head.
                          None (default) for regular ViTPose models.
                          0 = COCO, 1 = AiC, 2 = MPII, 3 = AP-10K, 4 = APT-36K, 5 = COCO-WholeBody
        """
        self.vitpose_model_name = vitpose_model_name
        self.person_detector_name = person_detector_name
        self._model_name = model_name
        self.dataset_index = dataset_index
        
        # Import here to avoid requiring transformers for other adapters
        from transformers import AutoProcessor, VitPoseForPoseEstimation, RTDetrForObjectDetection
        import torch
        
        self.torch = torch
        self.AutoProcessor = AutoProcessor
        self.VitPoseForPoseEstimation = VitPoseForPoseEstimation
        self.RTDetrForObjectDetection = RTDetrForObjectDetection
    
    def get_keypoint_mapping(self):
        """
        ViTPose uses COCO format (same as YOLO).
        17 keypoints indexed 0-16.
        
        NOTE: Like YOLO, ViTPose does NOT have foot_index keypoints,
        so we exclude them. ViTPose will be evaluated on 12 keypoints.
        """
        return {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16,
            # NOTE: No 'left_foot_index' or 'right_foot_index'
        }
    
    def run_inference(self, images_folder):
        """
        Run ViTPose inference on all frames.
        
        Process:
        1. Load RT-DETR person detector
        2. Load ViTPose model
        3. For each frame:
           a. Detect person with RT-DETR
           b. If person found, run ViTPose on bounding box
           c. Extract keypoints and wrap in our format
        
        Returns:
            predictions: dict {frame_num: KeypointWrapper or None}
            inference_time: float (seconds)
        """
        from PIL import Image
        
        # Load models
        print(f"  Loading person detector: {self.person_detector_name}")
        person_processor = self.AutoProcessor.from_pretrained(self.person_detector_name, use_fast=True)
        person_model = self.RTDetrForObjectDetection.from_pretrained(self.person_detector_name)
        
        print(f"  Loading pose estimator: {self.vitpose_model_name}")
        if self.dataset_index is not None:
            print(f"  Using dataset_index: {self.dataset_index}")
        pose_processor = self.AutoProcessor.from_pretrained(self.vitpose_model_name, use_fast=True)
        pose_model = self.VitPoseForPoseEstimation.from_pretrained(self.vitpose_model_name)
        
        # Move to GPU if available
        device = self.torch.device("cuda" if self.torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        person_model = person_model.to(device)
        pose_model = pose_model.to(device)
        
        predictions = {}
        frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.PNG')])
        
        start_time = time.time()
        
        for i, frame_file in enumerate(frame_files):
            frame_num = int(frame_file.split('_')[1].split('.')[0])
            
            # Print progress every 10 frames
            if i % 10 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / i
                    remaining = (len(frame_files) - i) * avg_time
                    print(f"  Progress: {i}/{len(frame_files)} frames ({i/len(frame_files)*100:.1f}%) - "
                          f"Avg: {avg_time:.2f}s/frame - ETA: {remaining/60:.1f}min")
            
            image_path = os.path.join(images_folder, frame_file)
            
            # Load image
            image = Image.open(image_path)
            img_width, img_height = image.size
            
            # Step 1: Detect person with RT-DETR
            person_inputs = person_processor(images=image, return_tensors="pt").to(device)
            
            with self.torch.no_grad():
                person_outputs = person_model(**person_inputs)
            
            # Post-process to get bounding boxes
            person_results = person_processor.post_process_object_detection(
                person_outputs, 
                target_sizes=self.torch.tensor([(img_height, img_width)]), 
                threshold=0.3
            )
            result = person_results[0]
            
            # Filter for person class (label 0 in COCO)
            person_boxes = result["boxes"][result["labels"] == 0]
            
            if len(person_boxes) == 0:
                # No person detected
                predictions[frame_num] = None
                continue
            
            # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
            person_boxes = person_boxes.cpu().numpy()
            person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]  # width
            person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]  # height
            
            # Pick best person (highest confidence or first if only one)
            if len(person_boxes) > 1:
                # Get confidences and pick best
                confidences = result["scores"][result["labels"] == 0].cpu().numpy()
                best_person_idx = confidences.argmax()
                person_box = person_boxes[best_person_idx:best_person_idx+1]
            else:
                person_box = person_boxes
            
            # Step 2: Run ViTPose on detected person
            pose_inputs = pose_processor(image, boxes=[person_box], return_tensors="pt").to(device)
            
            # For ViTPose++ models, pass dataset_index
            if self.dataset_index is not None:
                dataset_index_tensor = self.torch.tensor([self.dataset_index], device=device)
                with self.torch.no_grad():
                    pose_outputs = pose_model(**pose_inputs, dataset_index=dataset_index_tensor)
            else:
                # Regular ViTPose (no dataset_index)
                with self.torch.no_grad():
                    pose_outputs = pose_model(**pose_inputs)
            
            # Step 3: Post-process to get keypoints
            pose_results = pose_processor.post_process_pose_estimation(
                pose_outputs, 
                boxes=[person_box]
            )
            
            # Extract keypoints for first (and only) image
            if len(pose_results) > 0 and len(pose_results[0]) > 0:
                person_result = pose_results[0][0]  # First image, first person
                keypoints = person_result['keypoints']  # Shape: [17, 2]
                
                # Convert to numpy if it's a tensor
                if hasattr(keypoints, 'cpu'):
                    keypoints = keypoints.cpu().numpy()
                
                # Normalize to [0, 1] range (ViTPose returns absolute coordinates)
                keypoints[:, 0] = keypoints[:, 0] / img_width   # x coordinates
                keypoints[:, 1] = keypoints[:, 1] / img_height  # y coordinates
                
                # Wrap in our format
                predictions[frame_num] = ViTPoseKeypointWrapper(keypoints)
            else:
                # Pose estimation failed
                predictions[frame_num] = None
        
        total_time = time.time() - start_time
        
        return predictions, total_time
    
    def get_model_name(self):
        """
        Generate name from model path if not provided.
        
        Examples:
            'usyd-community/vitpose-base-simple' → 'ViTPose-Base-Simple'
            'usyd-community/vitpose-plus-huge' → 'ViTPose-Plus-Huge'
        """
        if self._model_name:
            return self._model_name
        
        # Extract model name from HuggingFace path
        model_part = self.vitpose_model_name.split('/')[-1]  # 'vitpose-base-simple'
        
        # Capitalize each part
        parts = model_part.split('-')
        formatted = '-'.join(p.capitalize() for p in parts)  # 'Vitpose-Base-Simple'
        
        return formatted
    

class ViTPoseKeypointWrapper:
    """
    Wraps ViTPose keypoint array to match MediaPipe interface.
    
    ViTPose gives us an array of shape [17, 2] (17 keypoints, x and y)
    We need to make it indexable and have .x and .y attributes.
    
    This is identical to YOLOKeypointWrapper but kept separate for clarity.
    """
    
    def __init__(self, vitpose_keypoints):
        """
        Args:
            vitpose_keypoints: numpy array of shape [17, 2]
        """
        self.keypoints = vitpose_keypoints
    
    def __getitem__(self, idx):
        """
        Make it indexable like MediaPipe landmarks.
        
        Example:
            kp = wrapper[5]  # Get left shoulder
            kp.x  # Access x coordinate
            kp.y  # Access y coordinate
        """
        kp = self.keypoints[idx]
        return KeypointWrapper(x=kp[0], y=kp[1])

class HRNetAdapter(PoseModelAdapter):
    """
    Adapter for HRNet pose estimation models.
    
    HRNet is a top-down approach that requires person detection first.
    We use RT-DETR for person detection, then HRNet for keypoint estimation.
    
    Based on: "Deep High-Resolution Representation Learning for Human Pose Estimation" (CVPR 2019)
    """
    
    def __init__(self, model_path, config_path, model_name=None,
                 person_detector_name="PekingU/rtdetr_r50vd_coco_o365"):
        """
        Args:
            model_path: Path to HRNet .pth file (e.g., 'pose_hrnet_w32_256x192.pth')
            config_path: Path to HRNet config YAML file
            model_name: Optional custom name for results (otherwise inferred from model_path)
            person_detector_name: HuggingFace model name for person detection
        """
        self.model_path = model_path
        self.config_path = config_path
        self.person_detector_name = person_detector_name
        self._model_name = model_name
        
        # Import HRNet dependencies
        import sys
        hrnet_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'hrnet')
        hrnet_root = os.path.abspath(hrnet_root)
        sys.path.insert(0, hrnet_root)
        sys.path.insert(0, os.path.join(hrnet_root, 'lib'))
        
        from config import cfg as hrnet_cfg
        import models as hrnet_models
        from core.inference import get_final_preds as hrnet_get_final_preds
        from utils.transforms import get_affine_transform as hrnet_get_affine_transform
        from transformers import RTDetrForObjectDetection, AutoProcessor
        
        # Store imports
        self.hrnet_cfg = hrnet_cfg
        self.hrnet_models = hrnet_models
        self.get_final_preds = hrnet_get_final_preds
        self.get_affine_transform = hrnet_get_affine_transform
        self.RTDetrForObjectDetection = RTDetrForObjectDetection
        self.AutoProcessor = AutoProcessor
    
    def get_keypoint_mapping(self):
        """
        HRNet uses COCO format (same as YOLO and ViTPose).
        17 keypoints indexed 0-16.
        
        NOTE: Like YOLO/ViTPose, HRNet does NOT have foot_index keypoints,
        so we exclude them. HRNet will be evaluated on 12 keypoints.
        """
        return {
            'left_shoulder': 5,
            'right_shoulder': 6,
            'left_elbow': 7,
            'right_elbow': 8,
            'left_wrist': 9,
            'right_wrist': 10,
            'left_hip': 11,
            'right_hip': 12,
            'left_knee': 13,
            'right_knee': 14,
            'left_ankle': 15,
            'right_ankle': 16,
            # NOTE: No 'left_foot_index' or 'right_foot_index'
        }
    
    def run_inference(self, images_folder):
        """
        Run HRNet inference on all frames.
        
        Process:
        1. Load RT-DETR person detector
        2. Load HRNet model
        3. For each frame:
           a. Detect person with RT-DETR
           b. If person found, compute center/scale from bbox
           c. Apply affine transform to crop person region
           d. Run HRNet on cropped region
           e. Extract keypoints and wrap in our format
        
        Returns:
            predictions: dict {frame_num: KeypointWrapper or None}
            inference_time: float (seconds)
        """
        import cv2
        import numpy as np
        from PIL import Image
        from torchvision import transforms
        
        # Load HRNet config
        self.hrnet_cfg.defrost()
        self.hrnet_cfg.merge_from_file(self.config_path)
        self.hrnet_cfg.freeze()
        
        # Load HRNet model
        print(f"  Loading HRNet model: {self.model_path}")
        hrnet_model = eval('self.hrnet_models.'+self.hrnet_cfg.MODEL.NAME+'.get_pose_net')(
            self.hrnet_cfg, is_train=False
        )
        state_dict = torch.load(self.model_path, map_location='cpu')
        hrnet_model.load_state_dict(state_dict)
        hrnet_model.eval()
        
        # Load person detector
        print(f"  Loading person detector: {self.person_detector_name}")
        detector_processor = self.AutoProcessor.from_pretrained(self.person_detector_name, use_fast=True)
        detector_model = self.RTDetrForObjectDetection.from_pretrained(self.person_detector_name)
        
        # Move to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        hrnet_model = hrnet_model.to(device)
        detector_model = detector_model.to(device)
        
        # Model parameters
        model_width = self.hrnet_cfg.MODEL.IMAGE_SIZE[0]
        model_height = self.hrnet_cfg.MODEL.IMAGE_SIZE[1]
        pixel_std = 200
        aspect_ratio = model_width * 1.0 / model_height
        
        # Preprocessing
        transform_fn = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225]),
        ])
        
        predictions = {}
        frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.PNG')])
        
        start_time = time.time()
        
        for i, frame_file in enumerate(frame_files):
            frame_num = int(frame_file.split('_')[1].split('.')[0])
            
            # Print progress every 10 frames
            if i % 10 == 0:
                elapsed = time.time() - start_time
                if i > 0:
                    avg_time = elapsed / i
                    remaining = (len(frame_files) - i) * avg_time
                    print(f"  Progress: {i}/{len(frame_files)} frames ({i/len(frame_files)*100:.1f}%) - "
                          f"Avg: {avg_time:.2f}s/frame - ETA: {remaining/60:.1f}min")
            
            image_path = os.path.join(images_folder, frame_file)
            
            # Load image
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_height, img_width = image.shape[:2]
            
            # Step 1: Detect person with RT-DETR
            pil_image = Image.fromarray(image_rgb)
            detector_inputs = detector_processor(images=pil_image, return_tensors="pt").to(device)
            
            with torch.no_grad():
                detector_outputs = detector_model(**detector_inputs)
            
            # Post-process to get bounding boxes
            detector_results = detector_processor.post_process_object_detection(
                detector_outputs, 
                target_sizes=torch.tensor([(img_height, img_width)]), 
                threshold=0.3
            )
            
            # Filter for person class (label 0 in COCO)
            person_boxes = detector_results[0]["boxes"][detector_results[0]["labels"] == 0]
            
            if len(person_boxes) == 0:
                # No person detected
                predictions[frame_num] = None
                continue
            
            # Pick best person (highest confidence or first if only one)
            if len(person_boxes) > 1:
                person_scores = detector_results[0]["scores"][detector_results[0]["labels"] == 0]
                best_idx = person_scores.argmax()
                bbox_xyxy = person_boxes[best_idx].cpu().numpy()
            else:
                bbox_xyxy = person_boxes[0].cpu().numpy()
            
            # Convert from xyxy to xywh format
            x = float(bbox_xyxy[0])
            y = float(bbox_xyxy[1])
            w = float(bbox_xyxy[2] - bbox_xyxy[0])
            h = float(bbox_xyxy[3] - bbox_xyxy[1])
            
            # Step 2: Compute center and scale using HRNet's method
            center = np.zeros((2), dtype=np.float32)
            center[0] = x + w * 0.5
            center[1] = y + h * 0.5
            
            # Adjust w/h to match model aspect ratio
            if w > aspect_ratio * h:
                h = w * 1.0 / aspect_ratio
            elif w < aspect_ratio * h:
                w = h * aspect_ratio
            
            scale = np.array([w * 1.0 / pixel_std, h * 1.0 / pixel_std], dtype=np.float32)
            
            # Step 3: Apply affine transformation to crop person region
            trans = self.get_affine_transform(center, scale, 0, [model_width, model_height])
            input_image = cv2.warpAffine(
                image_rgb,
                trans,
                (int(model_width), int(model_height)),
                flags=cv2.INTER_LINEAR
            )
            
            # Step 4: Normalize and convert to tensor
            input_tensor = transform_fn(input_image).unsqueeze(0).to(device)
            
            # Step 5: Run HRNet inference
            with torch.no_grad():
                output = hrnet_model(input_tensor)
            
            heatmaps = output.cpu().numpy()
            
            # Step 6: Post-process to get keypoints
            center_batch = center.reshape(1, -1)
            scale_batch = scale.reshape(1, -1)
            
            preds, maxvals = self.get_final_preds(
                self.hrnet_cfg, heatmaps, center_batch, scale_batch
            )
            
            # Step 7: Normalize coordinates to [0, 1]
            preds_normalized = preds.copy()
            preds_normalized[0, :, 0] = preds[0, :, 0] / img_width
            preds_normalized[0, :, 1] = preds[0, :, 1] / img_height
            
            # Wrap in our format
            predictions[frame_num] = HRNetKeypointWrapper(preds_normalized[0])
        
        total_time = time.time() - start_time
        
        return predictions, total_time
    
    def get_model_name(self):
        """
        Generate name from model path if not provided.
        
        Examples:
            'pose_hrnet_w32_256x192.pth' → 'HRNet-W32-256x192'
            'pose_hrnet_w48_384x288.pth' → 'HRNet-W48-384x288'
        """
        if self._model_name:
            return self._model_name
        
        # Extract from filename
        filename = os.path.basename(self.model_path)  # 'pose_hrnet_w32_256x192.pth'
        name_part = filename.replace('pose_', '').replace('.pth', '')  # 'hrnet_w32_256x192'
        
        # Parse components
        parts = name_part.split('_')
        if len(parts) >= 3:
            # hrnet, w32, 256x192
            model_type = parts[0].upper()  # HRNET
            width = parts[1].upper()       # W32
            resolution = parts[2]          # 256x192
            return f"{model_type}-{width}-{resolution}"
        
        # Fallback
        return name_part.replace('_', '-').title()


class HRNetKeypointWrapper:
    """
    Wraps HRNet keypoint array to match MediaPipe interface.
    
    HRNet gives us an array of shape [17, 2] (17 keypoints, x and y normalized)
    We need to make it indexable and have .x and .y attributes.
    """
    
    def __init__(self, hrnet_keypoints):
        """
        Args:
            hrnet_keypoints: numpy array of shape [17, 2] with normalized coordinates
        """
        self.keypoints = hrnet_keypoints
    
    def __getitem__(self, idx):
        """
        Make it indexable like MediaPipe landmarks.
        
        Example:
            kp = wrapper[5]  # Get left shoulder
            kp.x  # Access x coordinate
            kp.y  # Access y coordinate
        """
        kp = self.keypoints[idx]
        return KeypointWrapper(x=kp[0], y=kp[1])

class LightweightOpenPoseAdapter(PoseModelAdapter):
    """Adapter for Lightweight OpenPose (bottom-up, multi-person pose estimation)"""
    
    def __init__(self, model_path, model_name=None, repo_path='../models/lightweight_openpose'):
        self.model_path = model_path
        # Anchor path relative to this file, not the notebook's CWD
        repo_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'models', 'lightweight_openpose')
        self.repo_path = os.path.abspath(repo_path)
        self._model_name = model_name
        
        import sys
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
    
    def get_keypoint_mapping(self):
        """OpenPose uses 18 keypoints (0-17), missing foot_index like YOLO/ViTPose"""
        return {
            'left_shoulder': 5,
            'right_shoulder': 2,
            'left_elbow': 6,
            'right_elbow': 3,
            'left_wrist': 7,
            'right_wrist': 4,
            'left_hip': 11,
            'right_hip': 8,
            'left_knee': 12,
            'right_knee': 9,
            'left_ankle': 13,
            'right_ankle': 10,
        }
    
    def run_inference(self, images_folder):

        import sys

        # Evict any cached 'models' that ultralytics may have registered
        for key in list(sys.modules.keys()):
            if key == 'models' or key.startswith('models.'):
                del sys.modules[key]
        import cv2
        import numpy as np

        # Re-insert to be safe (idempotent)
        if self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)

        import cv2
        import numpy as np
        
        from models.with_mobilenet import PoseEstimationWithMobileNet
        from modules.keypoints import extract_keypoints, group_keypoints
        from modules.load_state import load_state
        from modules.pose import Pose
        from val import normalize, pad_width
        
        # Re-insert ultralytics models after our imports are done
        if self.repo_path in sys.path:
            sys.path.remove(self.repo_path)

        # Load model
        print(f"  Loading Lightweight OpenPose: {self.model_path}")
        net = PoseEstimationWithMobileNet()
        checkpoint = torch.load(self.model_path, map_location='cpu')
        load_state(net, checkpoint)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {device}")
        net = net.to(device)
        net.eval()
        
        # Parameters
        net_input_height_size = 256
        stride = 8
        upsample_ratio = 4
        num_keypoints = Pose.num_kpts
        img_mean = np.array([128, 128, 128], np.float32)
        img_scale = np.float32(1/256)
        
        predictions = {}
        frame_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.PNG')])
        
        start_time = time.time()
        
        for i, frame_file in enumerate(frame_files):
            frame_num = int(frame_file.split('_')[1].split('.')[0])
            
            if i % 10 == 0 and i > 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = (len(frame_files) - i) * avg_time
                print(f"  Progress: {i}/{len(frame_files)} ({i/len(frame_files)*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f}min")
            
            image_path = os.path.join(images_folder, frame_file)
            img = cv2.imread(image_path)
            height, width = img.shape[:2]
            
            # Preprocess
            scale = net_input_height_size / height
            scaled_img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled_img = normalize(scaled_img, img_mean, img_scale)
            min_dims = [net_input_height_size, max(scaled_img.shape[1], net_input_height_size)]
            padded_img, pad = pad_width(scaled_img, stride, (0, 0, 0), min_dims)
            
            tensor_img = torch.from_numpy(padded_img).permute(2, 0, 1).unsqueeze(0).float().to(device)
            
            # Inference
            with torch.no_grad():
                stages_output = net(tensor_img)
            
            # Extract heatmaps and PAFs
            stage2_heatmaps = stages_output[-2]
            heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
            heatmaps = cv2.resize(heatmaps, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
            
            stage2_pafs = stages_output[-1]
            pafs = np.transpose(stage2_pafs.squeeze().cpu().data.numpy(), (1, 2, 0))
            pafs = cv2.resize(pafs, (0, 0), fx=upsample_ratio, fy=upsample_ratio, interpolation=cv2.INTER_CUBIC)
            
            # Extract keypoints
            total_keypoints_num = 0
            all_keypoints_by_type = []
            for kpt_idx in range(num_keypoints):
                total_keypoints_num += extract_keypoints(heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)
            
            pose_entries, all_keypoints = group_keypoints(all_keypoints_by_type, pafs)
            
            if len(pose_entries) == 0:
                predictions[frame_num] = None
                continue
            
            # Transform to original coords
            for kpt_id in range(all_keypoints.shape[0]):
                all_keypoints[kpt_id, 0] = (all_keypoints[kpt_id, 0] * stride / upsample_ratio - pad[1]) / scale
                all_keypoints[kpt_id, 1] = (all_keypoints[kpt_id, 1] * stride / upsample_ratio - pad[0]) / scale
            
            # Build poses and select best
            best_pose = None
            best_confidence = -1
            
            for n in range(len(pose_entries)):
                if len(pose_entries[n]) == 0:
                    continue
                
                pose_keypoints = np.ones((num_keypoints, 2), dtype=np.float32) * -1
                for kpt_id in range(num_keypoints):
                    if pose_entries[n][kpt_id] != -1.0:
                        pose_keypoints[kpt_id, 0] = all_keypoints[int(pose_entries[n][kpt_id]), 0]
                        pose_keypoints[kpt_id, 1] = all_keypoints[int(pose_entries[n][kpt_id]), 1]
                
                confidence = pose_entries[n][18]
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_pose = pose_keypoints
            
            if best_pose is None:
                predictions[frame_num] = None
                continue
            
            # Normalize to [0, 1]
            best_pose[:, 0] = best_pose[:, 0] / width
            best_pose[:, 1] = best_pose[:, 1] / height
            
            predictions[frame_num] = LightweightOpenPoseKeypointWrapper(best_pose)
        
        total_time = time.time() - start_time
        return predictions, total_time
    
    def get_model_name(self):
        return self._model_name if self._model_name else "Lightweight-OpenPose"


class LightweightOpenPoseKeypointWrapper:
    """Wrapper for OpenPose keypoints to match MediaPipe interface"""
    
    def __init__(self, openpose_keypoints):
        self.keypoints = openpose_keypoints  # Shape: [18, 2], missing = negative values
    
    def __getitem__(self, idx):
        kp = self.keypoints[idx]  # Could be [0.68, 0.58] OR [-0.0009, -0.0005]
        
        if kp[0] < 0 or kp[1] < 0:  # ✓ Correctly detects missing
            return KeypointWrapper(x=0.0, y=0.0, confidence=0.0)  # ← NEW: Return clean zeros
        else:
            return KeypointWrapper(x=float(kp[0]), y=float(kp[1]), confidence=1.0)
        
class OpenPoseAdapter(PoseModelAdapter):
    """
    Adapter for OpenPose models (BODY_25, COCO, MPI).
    Uses the Windows portable binary via subprocess.
    """
    def __init__(self, model_type="BODY_25", openpose_dir="../models/openpose"):
        """
        Args:
            model_type: "BODY_25", "COCO", or "MPI"
            openpose_dir: Path to OpenPose installation folder
        """
        self.model_type = model_type
        self.openpose_dir = openpose_dir
        self.openpose_bin = os.path.join(openpose_dir, "bin", "OpenPoseDemo.exe")
        
        # Validate model type
        if model_type not in ["BODY_25", "COCO", "MPI"]:
            raise ValueError(f"Invalid model_type: {model_type}. Must be BODY_25, COCO, or MPI")
    
    def get_keypoint_mapping(self):
        """Return keypoint mapping based on model type."""
        
        if self.model_type == "BODY_25":
            # 25 keypoints including foot
            # Based on: {0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist, 
            #            5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip,
            #            10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
            #            15: REye, 16: LEye, 17: REar, 18: LEar, 19: LBigToe,
            #            20: LSmallToe, 21: LHeel, 22: RBigToe, 23: RSmallToe, 24: RHeel}
            return {
                'left_shoulder': 5,    # LShoulder
                'right_shoulder': 2,   # RShoulder
                'left_elbow': 6,       # LElbow
                'right_elbow': 3,      # RElbow
                'left_wrist': 7,       # LWrist
                'right_wrist': 4,      # RWrist
                'left_hip': 12,        # LHip
                'right_hip': 9,        # RHip
                'left_knee': 13,       # LKnee
                'right_knee': 10,      # RKnee
                'left_ankle': 14,      # LAnkle
                'right_ankle': 11,     # RAnkle
                'left_foot_index': 19, # LBigToe
                'right_foot_index': 22 # RBigToe
            }
        
        elif self.model_type == "COCO":
            # 18 keypoints (OpenPose COCO format with Neck)
            # Based on: {0: Nose, 1: Neck, 2: R-Sho, 3: R-Elb, 4: R-Wr,
            #            5: L-Sho, 6: L-Elb, 7: L-Wr, 8: R-Hip, 9: R-Knee,
            #            10: R-Ank, 11: L-Hip, 12: L-Knee, 13: L-Ank,
            #            14: R-Eye, 15: L-Eye, 16: R-Ear, 17: L-Ear}
            return {
                'left_shoulder': 5,    # L-Sho
                'right_shoulder': 2,   # R-Sho
                'left_elbow': 6,       # L-Elb
                'right_elbow': 3,      # R-Elb
                'left_wrist': 7,       # L-Wr
                'right_wrist': 4,      # R-Wr
                'left_hip': 11,        # L-Hip
                'right_hip': 8,        # R-Hip
                'left_knee': 12,       # L-Knee
                'right_knee': 9,       # R-Knee
                'left_ankle': 13,      # L-Ank
                'right_ankle': 10,     # R-Ank
                # NO foot_index
            }
        
        elif self.model_type == "MPI":
            # 15 keypoints (MPII format)
            # Based on: {0: r ankle, 1: r knee, 2: r hip, 3: l hip, 4: l knee,
            #            5: l ankle, 6: pelvis, 7: thorax, 8: upper neck, 9: head top,
            #            10: r wrist, 11: r elbow, 12: r shoulder, 13: l shoulder,
            #            14: l elbow, 15: l wrist}
            return {
                'left_ankle':   13,
                'left_knee':    12,
                'left_hip':     11,

                'right_ankle':  10,
                'right_knee':   9,
                'right_hip':    8,

                'left_elbow':   6,
                'right_elbow':  3,

                'left_shoulder': 5,     # best available match
                'right_shoulder': 2,

                'left_wrist':   7,   # no wrist detected in MPI model output
                'right_wrist':  4,
            }
    
    def run_inference(self, images_folder):
        """
        Run OpenPose on all frames in folder.

        Timing uses OpenPose's self-reported 'Total time' from stdout, which
        excludes subprocess startup overhead (~0.5s fixed cost) and is therefore
        comparable to the in-process timing used by other adapters.
        
        Returns:
            predictions: dict {frame_num: OpenPoseKeypointWrapper or None}
            inference_time: float (seconds)
        """
        import re
        import json
        import tempfile
        import shutil
        import subprocess
        from PIL import Image
        
        # Create temporary output directory
        temp_output = tempfile.mkdtemp(prefix=f"openpose_{self.model_type}_")
        
        try:
            cmd = [
                self.openpose_bin,
                "--image_dir", os.path.abspath(images_folder),
                "--write_json", temp_output,
                "--display", "0",
                "--render_pose", "0",
                "--model_pose", self.model_type
            ]
            
            print(f"  Running OpenPose ({self.model_type})...")
            
            wall_start = time.time()
            result = subprocess.run(
                cmd,
                cwd=self.openpose_dir,
                capture_output=True,
                text=True
            )
            wall_time = time.time() - wall_start

            if result.returncode != 0:
                print(f"  OpenPose error: {result.stderr}")
                return {}, wall_time

            # Use OpenPose's self-reported time — excludes subprocess startup overhead
            # and is therefore comparable to in-process timing used by other adapters.
            match = re.search(r'Total time:\s*([\d.]+)\s*seconds', result.stdout)
            if match:
                inference_time = float(match.group(1))
            else:
                print(f"  Warning: could not parse OpenPose timing, falling back to wall clock")
                inference_time = wall_time

            # Load all JSON results
            predictions = {}
            json_files = sorted([f for f in os.listdir(temp_output) if f.endswith('.json')])
            
            for json_file in json_files:
                frame_num = int(json_file.split('_')[1])
                
                frame_file = f"frame_{frame_num:06d}.PNG"
                image_path = os.path.join(images_folder, frame_file)
                img = Image.open(image_path)
                img_width, img_height = img.size

                json_path = os.path.join(temp_output, json_file)
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                if len(data['people']) > 0:
                    person = data['people'][0]
                    keypoints_flat = person['pose_keypoints_2d']
                    
                    num_keypoints = len(keypoints_flat) // 3
                    keypoints = []
                    for i in range(num_keypoints):
                        idx = i * 3
                        x    = keypoints_flat[idx]
                        y    = keypoints_flat[idx + 1]
                        conf = keypoints_flat[idx + 2]
                        keypoints.append([x / img_width, y / img_height, conf])
                    
                    predictions[frame_num] = OpenPoseKeypointWrapper(keypoints)
                else:
                    predictions[frame_num] = None
            
            print(f"  Processed {len(predictions)} frames in {inference_time:.2f}s "
                  f"(wall: {wall_time:.2f}s)")
            print(f"  Average: {inference_time/len(predictions):.4f}s per frame")
            
            return predictions, inference_time
        
        finally:
            if os.path.exists(temp_output):
                shutil.rmtree(temp_output)
    
    def get_model_name(self):
        """Return model name for results."""
        return f"OpenPose-{self.model_type}"


class OpenPoseKeypointWrapper:
    """
    Wraps OpenPose keypoint array to match MediaPipe interface.
    
    OpenPose gives us an array of shape [N, 3] where each row is [x, y, confidence]
    Coordinates are already normalized to [0, 1] range.
    """
    
    def __init__(self, keypoints):
        """
        Args:
            keypoints: list of [x_norm, y_norm, confidence] for each keypoint
        """
        self.keypoints = keypoints
    
    def __getitem__(self, idx):
        """
        Make it indexable like MediaPipe landmarks.
        
        Example:
            kp = wrapper[5]  # Get left shoulder
            kp.x  # Access x coordinate (normalized)
            kp.y  # Access y coordinate (normalized)
            kp.confidence  # Access confidence (0.0 if missing)
        """
        kp = self.keypoints[idx]
        return KeypointWrapper(x=kp[0], y=kp[1], confidence=kp[2])  # ← CHANGED: pass confidence