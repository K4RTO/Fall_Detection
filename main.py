import torch
import cv2
from torchvision import transforms
import numpy as np
import os
from tqdm import tqdm

from audio_algorithm import fall_detection_by_audio
from audio_algorithm.transform_video_and_audio import add_audio_to_video, extract_audio
from models import fall_detection_by_cv

from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utils.plots import output_to_keypoint


def get_frame_indices(vid_cap, time_scales):
    target_frames = []

    frame_count = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = vid_cap.get(cv2.CAP_PROP_FPS)

    if fps == 0:
        print("Error: Frame rate is zero, which is invalid.")
        return []
    if time_scales is None:
        return set()
    for time_scale in time_scales:
        start_frame = int(fps * time_scale[0])
        end_frame = int(fps * time_scale[1])
        if start_frame < frame_count and end_frame < frame_count:
            target_frames.append(list(range(start_frame, end_frame)))
        else:
            print(f"Warning: Time scale {time_scale} is out of video frame range.")

    return set(item for sublist in target_frames for item in sublist)


def falling_alarm_by_audio(image):
    height, width = image.shape[:2]
    thickness = int(min(height, width) * 0.06)  


    overlay = image.copy()

    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), thickness)


    text = "Fall detected by audio algorithm!"
    font_scale = 1.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - int(height * 0.1)  
    cv2.putText(overlay, text, (text_x, text_y), font, font_scale, (0, 0, 255), 2)

    return overlay  


def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 2550], thickness=3, lineType=cv2.LINE_AA)


def get_pose_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device: ", device)
    weigths = torch.load('yolov7-w6-pose.pt', map_location=device)
    model = weigths['model']
    _ = model.float().eval()
    if torch.cuda.is_available():
        model = model.half().to(device)
    return model, device


def get_pose(image, model, device):
    image = letterbox(image, 960, stride=64, auto=True)[0]
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))
    if torch.cuda.is_available():
        image = image.half().to(device)
    with torch.no_grad():
        output, _ = model(image)
    output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'],
                                     kpt_label=True)
    with torch.no_grad():
        output = output_to_keypoint(output)
    return image, output


def prepare_image(image):
    _image = image[0].permute(1, 2, 0) * 255
    _image = _image.cpu().numpy().astype(np.uint8)
    _image = cv2.cvtColor(_image, cv2.COLOR_RGB2BGR)
    return _image


def prepare_vid_out(directory, vid_cap, filename_with_suffix):
    vid_write_image = letterbox(vid_cap.read()[1], 960, stride=64, auto=True)[0]

    resize_height, resize_width = vid_write_image.shape[:2]

    out_video_name = f"{filename_with_suffix.split('.')[0]}_keypoint.mp4"
    out_path = os.path.join(directory, out_video_name)

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), vid_cap.get(cv2.CAP_PROP_FPS), (resize_width, resize_height), True)
    if not out.isOpened():
        print(f"Error: Cannot create video writer for {out_path}. Check codec support and path.")

    return out, out_path


def process_video(read_directory: str, write_directory: str, temp_directory: str, filename_with_suffix: str):
    current_video_path = os.path.join(read_directory, filename_with_suffix)
    output_path = os.path.join(write_directory, f"[Processed]{filename_with_suffix}")

    print(f"Processing video: {current_video_path}")

    vid_cap = cv2.VideoCapture(current_video_path)
    vid_out, temp_video_path = prepare_vid_out(temp_directory, vid_cap, filename_with_suffix)

    if not vid_cap.isOpened():
        print(f"Error: Failed to open video {current_video_path}")
        return

    # get pose estimation model
    model, device = get_pose_model()

    # make a list of original frames
    frames = []
    success, frame = vid_cap.read()
    if not success:
        print("Error: Failed to read the first frame.")
    while success:
        frames.append(frame)
        success, frame = vid_cap.read()

    # extract audio from video
    audio_save_directory = "data/temp/audio"
    if not os.path.exists(audio_save_directory):
        os.makedirs(audio_save_directory)
    audio_path = os.path.join(audio_save_directory, f"{filename_with_suffix.split('.')[0]}_audio.wav")
    extract_audio(current_video_path, audio_path)

    # get a set of frames that detected by audio algorithm
    time_scale = fall_detection_by_audio(audio_path, "fall_detection_model_by_audio.pth")
    target_frame_indices = get_frame_indices(vid_cap, time_scale)

    for index, image in enumerate(tqdm(frames)):
        image, output = get_pose(image, model, device)  # get current pose with pose model in current frame
        _image = prepare_image(image)
        is_fall, bbox = fall_detection_by_cv(output)  # test current pose with model
        if is_fall:
            falling_alarm(_image, bbox)
        if index in target_frame_indices:
            _image = falling_alarm_by_audio(_image)
        vid_out.write(_image)

    vid_out.release()
    vid_cap.release()

    add_audio_to_video(temp_video_path, audio_path, output_path)


def real_time_fall_detection():
    camera = cv2.VideoCapture(0)  # Initialize the camera

    # Check if the camera opened successfully
    if not camera.isOpened():
        print("Error: The camera could not be opened.")
        return

    model, device = get_pose_model()  # Load the pose detection model

    try:
        while True:
            success, frame = camera.read()  # Read a frame from the camera
            if not success:
                break  # If the frame is not successfully read, break out of the loop

            # Process the frame for pose detection and fall detection
            # image, output = get_pose(frame, model, device)
            # _image = prepare_image(image)

            # is_fall, bbox = fall_detection(output)
            #
            # if is_fall:
            #     falling_alarm(_image, bbox)
            l_image = frame

            # cv2.imshow("Fall Detection", _image)
            cv2.imshow("Camera View", l_image)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # real_time_fall_detection()

    directory_in = 'data/input/videos'
    directory_out = "data/output/videos"
    directory_temp = 'data/temp/videos'
    if not os.path.exists(directory_in):
        os.makedirs(directory_in)
    if not os.path.exists(directory_out):
        os.makedirs(directory_out)
    if not os.path.exists(directory_temp):
        os.makedirs(directory_temp)
    print(f"Available videos in {directory_in}:")
    for filename in os.listdir(directory_in):
        print(filename)
    print("---------------------------------------------")

    for filename in os.listdir(directory_in):
        file_in_path = os.path.join(directory_in, filename)
        if os.path.isfile(file_in_path):
            process_video(directory_in, directory_out, directory_temp, filename)
        else:
            print(f"{file_in_path} is not a file")


