# data_loading.py

import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from pathlib import Path

# Get project root (parent of src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def parse_cvat_annotations(xml_path):
    # Parse CVAT XML and return ground truth keypoints and head bboxes per frame
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    gt_keypoints = {}
    head_bboxes = {}
    
    # Parse skeleton tracks (person keypoints)
    for track in root.findall('.//track[@label="person"]'):
        for skeleton in track.findall('skeleton'):
            frame_num = int(skeleton.get('frame'))
            
            if frame_num not in gt_keypoints:
                gt_keypoints[frame_num] = {}
            
            for points in skeleton.findall('points'):
                label = points.get('label')
                coords = points.get('points')
                occluded = int(points.get('occluded', 0))
                
                x, y = map(float, coords.split(','))
                
                gt_keypoints[frame_num][label] = {
                    'x': x,
                    'y': y,
                    'occluded': occluded
                }
    
    # Parse head bounding boxes
    for track in root.findall('.//track[@label="head"]'):
        for box in track.findall('box'):
            frame_num = int(box.get('frame'))
            
            xtl = float(box.get('xtl'))
            ytl = float(box.get('ytl'))
            xbr = float(box.get('xbr'))
            ybr = float(box.get('ybr'))
            
            head_bboxes[frame_num] = {
                'xtl': xtl,
                'ytl': ytl,
                'xbr': xbr,
                'ybr': ybr,
                'width': xbr - xtl,
                'height': ybr - ytl
            }
    
    return gt_keypoints, head_bboxes


def split_frames_into_videos(gt_keypoints, head_bboxes, frames_per_video=24):
    # Split frames into videos (24 frames each)
    videos = {}
    
    all_frames = sorted(gt_keypoints.keys())
    num_videos = len(all_frames) // frames_per_video
    
    for video_id in range(num_videos):
        start_frame = video_id * frames_per_video
        end_frame = start_frame + frames_per_video
        
        video_frames = range(start_frame, end_frame)
        
        videos[video_id] = {
            'keypoints': {f: gt_keypoints[f] for f in video_frames if f in gt_keypoints},
            'head_bboxes': {f: head_bboxes[f] for f in video_frames if f in head_bboxes},
            'frames': list(video_frames)
        }
    
    return videos

def organize_predictions_by_video(predictions, videos):
    # Organize model predictions by video
    video_predictions = {}
    
    for video_id, video_data in videos.items():
        video_predictions[video_id] = {}
        for frame_num in video_data['frames']:
            if frame_num in predictions:
                video_predictions[video_id][frame_num] = predictions[frame_num]
    
    return video_predictions

def get_image_dimensions(frame_num):
    # Get actual image dimensions for a frame
    img_path = PROJECT_ROOT / 'data' / 'annotations' / 'images' / 'default' / f'frame_{frame_num:06d}.PNG'
    img = Image.open(img_path)
    return img.size  # Returns (width, height)

def detect_visible_side(video_keypoints):
    # Detect which side (left/right) is visible based on occlusion
    left_occluded = 0
    right_occluded = 0
    
    for frame_kps in video_keypoints.values():
        for kp_name, kp_data in frame_kps.items():
            if 'left_' in kp_name and kp_data['occluded'] == 1:
                left_occluded += 1
            elif 'right_' in kp_name and kp_data['occluded'] == 1:
                right_occluded += 1
    
    return 'right' if left_occluded > right_occluded else 'left'

def calculate_leg_length(keypoints, frame_num, visible_side):
    # Calculate leg length from hip to ankle
    frame_kps = keypoints[frame_num]
    
    hip_key = f'{visible_side}_hip'
    knee_key = f'{visible_side}_knee'
    ankle_key = f'{visible_side}_ankle'
    
    if hip_key not in frame_kps or knee_key not in frame_kps or ankle_key not in frame_kps:
        return None
    
    hip = frame_kps[hip_key]
    knee = frame_kps[knee_key]
    ankle = frame_kps[ankle_key]
    
    if hip['occluded'] or knee['occluded'] or ankle['occluded']:
        return None
    
    hip_knee = np.sqrt((hip['x'] - knee['x'])**2 + (hip['y'] - knee['y'])**2)
    knee_ankle = np.sqrt((knee['x'] - ankle['x'])**2 + (knee['y'] - ankle['y'])**2)
    
    return hip_knee + knee_ankle

def calculate_head_size(head_bbox):
    # Calculate head size as max(width, height) for PCKh normalization
    return max(head_bbox['width'], head_bbox['height'])