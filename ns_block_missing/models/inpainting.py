import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

from numpy import ndarray
from typing import Literal

__all__ = ['video_inpainting']


def navier_stokes_inpainting(video: ndarray, mask: ndarray) -> ndarray:
    # video.shape == (batch_size, n_channel, n_frame, image_size, image_size)
    batch_size, _, n_frames, _, _ = video.shape
    result = video.copy()
    
    for b in range(batch_size):
        for f in range(n_frames):
            frame = video[b, :, f, :, :].transpose(1, 2, 0)
            frame_mask = mask[b, :, f, :, :].transpose(1, 2, 0)
            
            binary_mask = ((1 - frame_mask[:, :, 0]) * 255).astype(np.uint8)
            
            if np.sum(binary_mask) == 0:
                continue
                
            frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
            
            repaired = cv2.inpaint(
                frame_uint8,
                binary_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_NS
            )
            
            repaired = repaired.astype(np.float32) / 255.0
            if repaired.ndim == 2:
                repaired = repaired[:, :, None]
            repaired_tensor = repaired.transpose(2, 0, 1)
            
            mask_b_f = mask[b, :, f, :, :]
            result[b, :, f, :, :] = mask_b_f * video[b, :, f, :, :] + (1 - mask_b_f) * repaired_tensor
    
    return result

def fast_marching_inpainting(video: ndarray, mask: ndarray) -> ndarray:
    # video.shape == (batch_size, n_channel, n_frame, image_size, image_size)
    batch_size, _, n_frames, _, _ = video.shape
    result = video.copy()

    for b in range(batch_size):
        for f in range(n_frames):
            frame = video[b, :, f, :, :].transpose(1, 2, 0)
            frame_mask = mask[b, :, f, :, :].transpose(1, 2, 0)
            
            binary_mask = ((1 - frame_mask[:, :, 0]) * 255).astype(np.uint8)
            
            if np.sum(binary_mask) == 0:
                continue
                
            frame_uint8 = np.clip(frame * 255, 0, 255).astype(np.uint8)
            
            repaired = cv2.inpaint(
                frame_uint8,
                binary_mask,
                inpaintRadius=3,
                flags=cv2.INPAINT_TELEA
            )
            
            repaired = repaired.astype(np.float32) / 255.0
            if repaired.ndim == 2:
                repaired = repaired[:, :, None]
            repaired_tensor = repaired.transpose(2, 0, 1)
            
            mask_b_f = mask[b, :, f, :, :]
            result[b, :, f, :, :] = mask_b_f * video[b, :, f, :, :] + (1 - mask_b_f) * repaired_tensor
    
    return result

def temporal_consistency_inpainting(video: ndarray, mask: ndarray) -> ndarray:
    batch_size, n_channel, n_frames, h, w = video.shape
    result = video.copy()
    
    for b in range(batch_size):
        for f in range(n_frames):
            current_mask = mask[b, :, f, :, :]
            
            if np.all(current_mask == 1):
                continue

            damaged_pixels = (current_mask == 0)
            
            if f > 0:
                prev_mask = mask[b, :, f-1, :, :]
                prev_frame = result[b, :, f-1, :, :]
                valid_from_prev = damaged_pixels & (prev_mask == 1)
                result[b, :, f, :, :] = np.where(valid_from_prev, prev_frame, result[b, :, f, :, :])
                
                damaged_pixels = damaged_pixels & ~valid_from_prev
            
            if f < n_frames - 1:
                next_mask = mask[b, :, f+1, :, :]
                next_frame = video[b, :, f+1, :, :]
                valid_from_next = damaged_pixels & (next_mask == 1)
                result[b, :, f, :, :] = np.where(valid_from_next, next_frame, result[b, :, f, :, :])
                
                damaged_pixels = damaged_pixels & ~valid_from_next
            
            for c in range(n_channel):
                if np.any(damaged_pixels[c]):
                    frame_channel = result[b, c, f, :, :]
                    mask_channel = (damaged_pixels[c] == 0).astype(np.float32)
                    
                    distances = distance_transform_edt(mask_channel)
                    
                    if np.any(damaged_pixels[c]):
                        weights = np.exp(-distances / 5.0)
                        weights[damaged_pixels[c]] = 0
                        
                        for i in range(h):
                            for j in range(w):
                                if damaged_pixels[c, i, j]:
                                    window_size = 5
                                    i_start = max(0, i - window_size)
                                    i_end = min(h, i + window_size + 1)
                                    j_start = max(0, j - window_size)
                                    j_end = min(w, j + window_size + 1)
                                    
                                    window_weights = weights[i_start:i_end, j_start:j_end]
                                    window_values = frame_channel[i_start:i_end, j_start:j_end]
                                    
                                    if np.sum(window_weights) > 0:
                                        result[b, c, f, i, j] = np.sum(window_weights * window_values) / np.sum(window_weights)
    
    return result


def video_inpainting(
        video: ndarray, mask: ndarray,
        method: Literal[
            'navier_stokes',
            'fast_marching',
            'temporal_consistency',
            'optical_flow'
        ],
        **kwargs
    ) -> ndarray:
    assert video.ndim == mask.ndim == 5
    assert video.shape == mask.shape

    match method:
        case 'navier_stokes':
            return navier_stokes_inpainting(video, mask)
        case 'fast_marching':
            return fast_marching_inpainting(video, mask)
        case 'temporal_consistency':
            return temporal_consistency_inpainting(video, mask)
        case 'optical_flow':
            raise NotImplementedError()
        case _:
            raise ValueError()
        

