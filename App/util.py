from pytube import YouTube
def pytube_dl(url):
    yt = YouTube(str(url))
    video = yt.streams.filter(file_extension='mp4').first()
    video.download('App/video/')

from App.utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
# from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages
import torch
from tqdm.autonotebook import tqdm
from App.utils.torch_utils import select_device, time_sync
from App.utils.filtering import filter_output
from App.utils.pdf_generator import pdf_generator, normalize
from App.utils.general import (check_file, check_img_size, cv2, non_max_suppression)

def prediction_pdf():
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='App/weights/best.pt')

    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size((640, 640), s=stride)  

    dataset = LoadImages("App/video/", img_size=imgsz, stride=stride, auto=pt, only_vids=True)
    bs = 1
    brand_count = {}
    path, im, im0s, vid_cap, s, frame = dataset.__iter__().__next__()

    framerate = 1
    initial_framerate = vid_cap.get(cv2.CAP_PROP_FPS)
    real_framerate = initial_framerate / round(initial_framerate / framerate)
    total_frames = dataset.frames
    dataset.frame = 0

    total_frames = dataset.frames
    device = select_device('')
    dt, seen = [0.0, 0.0, 0.0], 0

    save_dir = "App/saving_predict"
    save_unprocessed_output = False
    pred_timing_start = time_sync()

    for path, im, im0s, vid_cap, s, frame in tqdm(dataset, total=total_frames):

            # skip the frame if it isn't in the specified framerate
            if frame % round(initial_framerate / framerate) != 0:
                continue

            t1 = time_sync()
            im = torch.from_numpy(im).to(device)
            im = im.float()
            im /= 255  # 0 - 255 to 0.0 - 1.0

            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            t2 = time_sync()
            dt[0] += t2 - t1

            # Inference
            pred = model(im)
            t3 = time_sync()
            dt[1] += t3 - t2

            # NMS
            pred = non_max_suppression(pred, conf_thres=0.35, max_det=5)
            pred = pred[0].tolist()
            dt[2] += time_sync() - t3

            has_prediction = len(pred)
            if has_prediction:
                for brand in pred:
                    label = names[int(brand[5])]

                    # Retrieve or create a dictionnary key for the label and add the bbox, confidence and frame of the prediction
                    brand_count[label] = brand_count.get(label, {"bbox": [], "confidence": [], "frame": []})
                    brand_count[label]["bbox"].append(brand[0:4])
                    brand_count[label]["confidence"].append(brand[4])
                    brand_count[label]["frame"].append(frame)
        
        # Generate an output if a prediction has been made
    if brand_count:
        filtered_output = filter_output(brand_count, framerate)
        pdf_generator(path, filtered_output, save_dir)
        # cv2.imshow(im, framerate)

        if save_unprocessed_output:
            with open(f"{save_dir}/{normalize(path)}.txt", "w") as f:
                f.write(str(brand_count))
    else:
        print("No prediction has been made")


    pred_timing_stop = time_sync()
    pred_timing = pred_timing_stop - pred_timing_start
    print("Pred took %.2fs (%.2ffps)" % (pred_timing, ((total_frames / initial_framerate) * real_framerate) / pred_timing))