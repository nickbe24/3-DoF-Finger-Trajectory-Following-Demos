# this script takes in a video of the fingertip and outputs a video containing the mask, the coords of the fingertip 
# at each point, and a png of the first frame. I have little experience using pytorch and cv2 so gpt4 was used for help.

import cv2
import numpy as np
import torch
from segment_anything import sam_model_registry, SamPredictor

model_type = "vit_b"
checkpoint_path = "sam_vit_b_01ec64.pth"

# modify for video 
video_path = "step_set_01/step_5mms_02.mp4"
output_path_txt = "step_set_01/step_5mms_02_coords.txt"
output_path_video = "step_set_01/step_5mms_02_mask.mp4"
first_frame_path = "step_set_01/step_5mms_02_first_frame.png"

# load sam2
sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
sam.to("cuda" if torch.cuda.is_available() else "cpu")
predictor = SamPredictor(sam)

# open the video
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError("Failed to open video file")

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(output_path_video, fourcc, fps, (width, height))

# save first frame to use for pix / mm conversion
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read first frame")

cv2.imwrite(first_frame_path, first_frame)
print(f"First frame saved to {first_frame_path}")

# prompt to click for first sam seed

click_coords = [] 
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        click_coords.clear()
        click_coords.append((x, y))
        temp = first_frame.copy()
        cv2.circle(temp, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click to select object", temp)

cv2.imshow("Click to select object", first_frame)
cv2.setMouseCallback("Click to select object", click_event)
cv2.waitKey(0)
cv2.destroyAllWindows()

if not click_coords:
    raise RuntimeError("No click point selected.")

# helpers
def get_centroid(mask): # calc centroid of the mask - to use as seed for next frame
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)))

def get_leftmost_top_bottom_midpoint(mask): # use for fingertip position - may have to modify based on orientation over path
    """Return midpoint between topmost and leftmost points on the contour,
    plus those two points for drawing. Robust to rotations."""
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, None

    cnt = max(contours, key=cv2.contourArea)  
    cnt_pts = cnt[:, 0, :]  # (N,2)

    if cnt_pts.shape[0] < 2:
        return None, None, None

    def midpoint(a, b):
        return (int((a[0] + b[0]) * 0.5), int((a[1] + b[1]) * 0.5))

   
    topmost = cnt_pts[np.argmin(cnt_pts[:, 1])]
    leftmost = cnt_pts[np.argmin(cnt_pts[:, 0])]

    mid = midpoint(topmost, leftmost)
    return tuple(mid), tuple(topmost), tuple(leftmost)

input_point = np.array([click_coords[0]])
input_label = np.array([1])

predictor.set_image(first_frame)
masks, scores, _ = predictor.predict(point_coords=input_point, point_labels=input_label, multimask_output=True)
best_mask = masks[np.argmax(scores)]

centroid = get_centroid(best_mask)
edge_midpoint, p1, p2 = get_leftmost_top_bottom_midpoint(best_mask)
if not centroid or not edge_midpoint:
    raise RuntimeError("Failed to get initial tracking points.")

overlay = first_frame.copy()
overlay[best_mask] = [0, 255, 0]
cv2.circle(overlay, centroid, 4, (0, 0, 255), -1)      # red dot - centroid - used for prompting
cv2.circle(overlay, edge_midpoint, 4, (255, 0, 0), -1)  # blue dot - fingertip loc
cv2.line(overlay, p1, p2, (255, 255, 0), 2)             # cyan - edge
video_writer.write(overlay)

tracked_edge_points = [edge_midpoint]
frame_idx = 1
cap.set(cv2.CAP_PROP_POS_FRAMES, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    predictor.set_image(frame)

    # use centroid of previous mask as seed
    input_point = np.array([centroid])
    input_label = np.array([1])

    try:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True
        )
    except Exception as e:
        print(f"Frame {frame_idx}: SAM error → {e}")
        frame_idx += 1
        continue

    if masks is None or len(masks) == 0:
        print(f"Frame {frame_idx}: No masks.")
        frame_idx += 1
        continue

    best_mask = masks[np.argmax(scores)]
    centroid = get_centroid(best_mask)
    edge_midpoint, p1, p2 = get_leftmost_top_bottom_midpoint(best_mask)

    if centroid and edge_midpoint:
        tracked_edge_points.append(edge_midpoint)

    overlay = frame.copy()
    overlay[best_mask] = [0, 255, 0]
    if centroid:
        cv2.circle(overlay, centroid, 4, (0, 0, 255), -1)
    if edge_midpoint:
        cv2.circle(overlay, edge_midpoint, 4, (255, 0, 0), -1)
    if p1 and p2:
        cv2.line(overlay, p1, p2, (255, 255, 0), 2)
    video_writer.write(overlay)

    print(f"Frame {frame_idx+1}/{frame_count}", end="\r")
    frame_idx += 1

cap.release()
video_writer.release()
cv2.destroyAllWindows()

# save tracked points and video 
if tracked_edge_points:
    np.savetxt(output_path_txt, np.array(tracked_edge_points), fmt="%d")
    print(f"\nTracking complete.\nEdge path saved to {output_path_txt}\nVideo saved to {output_path_video}")
else:
    print("\nTracking failed — no valid edge points saved.")
