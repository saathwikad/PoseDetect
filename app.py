import torch
import cv2
import argparse
from flask import Flask, render_template, Response
import posenet
from posenet.utils import draw_boxes
import alarm

app = Flask(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()

# Load PoseNet model
model = posenet.load_model(args.model)
output_stride = model.output_stride

# Initialize camera capture object
camera = cv2.VideoCapture(args.cam_id)
camera.set(3, args.cam_width)
camera.set(4, args.cam_height)

def get_bounding_box(keypoint_coords, scale_factor=1.0, y_offset=0):
    min_x = min(keypoint_coords[:, 1])
    max_x = max(keypoint_coords[:, 1])
    min_y = min(keypoint_coords[:, 0])
    max_y = max(keypoint_coords[:, 0])
    
    # Calculate width and height of the bounding box
    width = max_x - min_x
    height = max_y - min_y
    
    # Calculate the center of the bounding box
    center_x = min_x + width / 2
    center_y = min_y + height / 2
    
    # Apply the scaling factor to width and height
    width *= scale_factor
    height *= scale_factor
    
    # Calculate new min_x and min_y based on the scaled width and height
    new_min_x = int(center_x - width / 2)
    new_min_y = int(center_y - height / 2) + y_offset  # Apply vertical offset
    
    return (new_min_x, new_min_y, int(width), int(height))

def check_head_within_boundary(correct_pos, current_pos, pad_x, pad_y):
    cx, cy, cw, ch = correct_pos
    x, y, w, h = current_pos

    return (x >= cx - pad_x and y >= cy - pad_y and
            x + w <= cx + cw + pad_x and y + h <= cy + ch + pad_y)


def gen_frames():
    my_alarm = alarm.Alarm(r"C:\Users\dsaat\Downloads\posenet-pytorch-master\posenet-pytorch-master\data\audio\alarm_audio.wav") 
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            input_image, display_image, output_scale = posenet.read_cap(
                camera, scale_factor=args.scale_factor, output_stride=output_stride)

            with torch.no_grad():
                input_image = torch.Tensor(input_image)
                heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

                pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                    heatmaps_result.squeeze(0),
                    offsets_result.squeeze(0),
                    displacement_fwd_result.squeeze(0),
                    displacement_bwd_result.squeeze(0),
                    output_stride=output_stride,
                    max_pose_detections=10,
                    min_pose_score=0.15)

            keypoint_coords *= output_scale

            if pose_scores[0] > 0.15:
                current_pos = get_bounding_box(keypoint_coords[0], scale_factor=0.8, y_offset=-20)
            else:
                current_pos = (0, 0, 0, 0)

            overlay_image = posenet.draw_skel_and_kp(
                display_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.15, min_part_score=0.1)

            pad_x = 30
            pad_y = 30
            img_width = args.cam_width
            img_height = args.cam_height

            center_x = img_width // 2
            center_y = img_height // 2
            bbox_width = 200
            bbox_height = 200

            top_left_x = center_x - (bbox_width // 2)
            top_left_y = center_y - (bbox_height // 2)

            correct_pos = (top_left_x, top_left_y, bbox_width, bbox_height)


            overlay_image = draw_boxes(overlay_image, correct_pos, current_pos, pad_x, pad_y)

            # Display bounding box coordinates
            box_text = f'Coordinates: {current_pos}'
            cv2.putText(overlay_image, box_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if current_pos != (0, 0, 0, 0):  # Replace with your condition to trigger the alarm
                if not my_alarm.is_playing():
                    my_alarm.play()
            else:
                if my_alarm.is_playing():
                    my_alarm.stop()

            ret, buffer = cv2.imencode('.jpg', overlay_image)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

# Release camera capture object
camera.release()
