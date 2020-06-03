import time

from pydarknet import Detector, Image
import cv2

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Process a video.')
    parser.add_argument('path', metavar='video_path', type=str,
                        help='Path to source video')
    parser.add_argument('yolo', metavar='yolo_tiny', type=int, choices=range(0, 2), 
			default = 1, 
                        help='Path to source video')

    args = parser.parse_args()
    print("Source Path:", args.path)
    cap = cv2.VideoCapture(args.path)

    average_time = 0

    if args.yolo == 0:
        net = Detector(bytes("cfg/yolov3.cfg", encoding="utf-8"),
                   bytes("weights/yolov3.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))
    else:
        net = Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"),
                   bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0,
                   bytes("cfg/coco.data", encoding="utf-8"))
    _, image = cap.read()
    h, w = image.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    out = cv2.VideoWriter("/content/YOLO3-4-Py/output.avi", fourcc, 20.0, (w, h))

    start_time_full = time.time()
    while True:
        r, frame = cap.read()
        try:
            h, w = frame.shape[:2]
        except:
            break
        if r:
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead

            dark_frame = Image(frame)
            results = net.detect(dark_frame)
            del dark_frame

            end_time = time.time()
            average_time = average_time * 0.8 + (end_time - start_time) * 0.2
            # Frames per second can be calculated as 1 frame divided by time required to process 1 frame
            fps = 1 / (end_time - start_time)

            print("FPS: ", fps)
            print("Total Time:", end_time - start_time, ":", average_time)

            for cat, score, bounds in results:
                catDec = str(cat.decode("utf-8"))
                if (catDec == "person"):
                    x, y, w, h = bounds
                    cv2.rectangle(frame, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)),
                                  (255, 0, 0))
                    cv2.putText(frame, catDec, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0))

            out.write(frame)
            # cv2_imshow(frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
    out.release()
end_time_full = time.time()
print("Total Time:", end_time_full - start_time_full)
