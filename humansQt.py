import cv2
import numpy as np

import time
import sys
from pydarknet import Detector, Image

from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit, QLabel, QPushButton, QProgressBar, \
    QRadioButton, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
font_scale = 1
thickness = 1
labels = open("data/coco.names").read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

class MyWidget(QWidget):
    def __init__  (self):
        QWidget. __init__ (self)
    myclose = True

    def closeEvent(self,event):
        if self.myclose:
            print(self.myclose)
            try:
                cap.release()
                cv2.destroyAllWindows()
            except:
                print("")
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MyWidget()
    w.resize(310,210)
    w.setWindowTitle('Humans')

    pbar = QProgressBar(w)
    pbar.setGeometry(10, 165, 290, 30)

    def progBarUpdate(percent):
        pbar.setValue(percent)

    radiobuttonYolo = QRadioButton("yolov3", w)
    radiobuttonYoloTiny = QRadioButton("yolov3-tiny", w)
    radiobuttonYolo.move(10,70)
    radiobuttonYoloTiny.move(10,100)
    radiobuttonYoloTiny.setChecked(True)
    radiobuttonYolo.show()
    radiobuttonYoloTiny.show()

    font_scale = 1
    thickness = 1

    def startRec():
        global cap
        if radiobuttonYoloTiny.isChecked():
            config_path = "cfg/yolov3-tiny.cfg"
            weights_path = "weights/yolov3-tiny.weights"
        else:
            config_path = "cfg/yolov3.cfg"
            weights_path = "weights/yolov3.weights"
        net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
        ln = net.getLayerNames()
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #now = datetime.now()

        starting_time = time.time()
        frame_id = 0

        video_file = nameEdit.text()
        try:
            cap = cv2.VideoCapture(video_file)
            nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            net = Detector(bytes("cfg/yolov3-tiny.cfg", encoding="utf-8"), bytes("weights/yolov3-tiny.weights", encoding="utf-8"), 0, bytes("cfg/coco.data", encoding="utf-8"))
            _, image = cap.read()
            h, w = image.shape[:2]
            #fourcc = cv2.VideoWriter_fourcc(*"XVID")
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
            out = cv2.VideoWriter("output.avi", fourcc, 20.0, (w, h))
            k = 0
            while True:
                _, image = cap.read()
                try:
                    h, w = image.shape[:2]
                except:
                    progBarUpdate(100)
                    break
                frame_id += 1
                start = time.perf_counter()
                dark_frame = Image(image)
                results = net.detect(dark_frame)
                del dark_frame
                time_took = time.perf_counter() - start
                
                progBarUpdate(100 * (k / nb_frames))
                k += 1
                print("Time took:", time_took, k, "/", nb_frames)
                
                for cat, score, bounds in results:
                    catDec = str(cat.decode("utf-8"))
                    text = f"{catDec}: {score:.2f}"
                    print(text)
                    if (catDec == "person"):
                        x, y, w, h = bounds
                        cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(0,118,255))
                        cv2.putText(image, text, (int(x), int(y)), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 128))

                out.write(image)
                cv2.imshow("image", image)

                if ord("q") == cv2.waitKey(1):
                    break

            cap.release()
            cv2.destroyAllWindows()
            #now1 = datetime.now()
            #print(now1-now)
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText("Файл недоступен")
            msg.setInformativeText("Проверьте корректность вводимого пути")
            msg.setWindowTitle("Ошибка чтения")
            msg.exec_()

    dirLabel = QLabel(w)
    dirLabel.setText("Расположение видеофайла:")
    dirLabel.move(10,10)
    dirLabel.show()

    nameEdit = QLineEdit(w)
    nameEdit.move(10,40)
    nameEdit.show()

    button = QPushButton(w)
    button.setText('Обработать')
    button.move(10,130)
    button.show()
    button.clicked.connect(startRec)

    w.show()
    sys.exit(app.exec_())
