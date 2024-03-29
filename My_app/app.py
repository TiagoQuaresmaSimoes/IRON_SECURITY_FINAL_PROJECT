from flask import Flask, render_template, Response, redirect, send_file, request
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
from threading import Thread
import os
import winsound
import threading
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__)

# Global variables to control camera functions
security_mode = False
camera_running = False
record_folder = ' ' #Records folder path

# Function to capture frames from regular camera
def reg_cam():
    model = YOLO("yolov8n-seg.pt")  # segmentation model
    names = model.model.names
    cap = cv2.VideoCapture(0)
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    while True:
        ret, im0 = cap.read()
        if not ret:
            print("Not working")
            break

        results = model.predict(im0)
        annotator = Annotator(im0, line_width=2)

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            for mask, cls in zip(masks, clss):
                annotator.seg_bbox(mask=mask,
                                   mask_color=colors(int(cls), True),
                                   det_label=names[int(cls)])

        ret, buffer = cv2.imencode('.jpg', im0)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()

# Function to capture frames from security camera
def sec_cam():
    video = cv2.VideoCapture(0)
    modelo = YOLO('yolov8n.pt')

#Security area
    #area = [100, 190, 1150, 700] #Big 
    #area = [110, 115, 700, 1000] #Hall
    area = [510, 230, 910, 700] #Small

    
    recording = {'value': False}
    out = {'value': None}
    start_time = None
    end_time = None

    password = "***************"  # Your decrypted password from gmail here
    from_email = "*******************"
    to_email = "******************"

    server = smtplib.SMTP('smtp.gmail.com: 587')
    server.starttls()
    server.login(from_email, password)

    def send_email(to_email, from_email, object_detected=1):
        message = MIMEMultipart()
        message['From'] = from_email
        message['To'] = to_email
        message['Subject'] = "Security Alert"
        message_body = f'ALERT - {object_detected} individual(s) have been detected!!'
        message.attach(MIMEText(message_body, 'plain'))
        server.sendmail(from_email, to_email, message.as_string())
        server.quit()
        
    def alarme():
        for _ in range(7):
            winsound.Beep(2500, 500)

    def start_recording(img):
        global record_folder
        recording['value'] = True
        start_time = time.time()
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        filename = os.path.join(record_folder, f"REC_{time.strftime('%Y-%m-%d_%H-%M-%S')}.mp4")
        out['value'] = cv2.VideoWriter(filename, fourcc, 10, (img.shape[1], img.shape[0]))

    def stop_recording():
        recording['value'] = False
        end_time = time.time()
        out['value'].release()

    def generate_frames():

        alarmeCtl = False

        while True:
            check, img = video.read()
            if not check:
                break

            img = cv2.resize(img, (1270, 720))
            img2 = img.copy()
            cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 255, 0), -1)
            resultado = modelo(img)

            object_detected = False

            for objetos in resultado:
                obj = objetos.boxes
                for dados in obj:
                    x, y, w, h = dados.xyxy[0]
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    cls = int(dados.cls[0])
                    cx, cy = (x + w) // 2, (y + h) // 2
                    if cls == 0:
                        cv2.rectangle(img, (x, y), (w, h), (255, 0, 0), 5)

                        if cx >= area[0] and cx <= area[2] and cy >= area[1] and cy <= area[3]:
                            cv2.rectangle(img2, (area[0], area[1]), (area[2], area[3]), (0, 0, 255), -1)
                            cv2.rectangle(img, (100, 30), (470, 80), (0, 0, 255), -1)
                            cv2.putText(img, 'INTRUDER DETECTED', (105, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
                            object_detected = True
                            if not alarmeCtl:
                                alarmeCtl = True
                                threading.Thread(target=alarme).start()

                            if not recording['value']:
                                threading.Thread(target=start_recording, args=(img,)).start()

                            threading.Thread(target=send_email, args=(to_email, from_email, len(obj))).start()

                    else:
                        object_detected = False

            if recording['value'] and object_detected:
                end_time = time.time()

            if recording['value'] and not object_detected and end_time is not None and time.time() - end_time >= 10:
                threading.Thread(target=stop_recording).start()

            if recording['value'] and out['value'] is not None:
                out['value'].write(img)

            imgFinal = cv2.addWeighted(img2, 0.5, img, 0.5, 0)

            ret, buffer = cv2.imencode('.jpg', imgFinal)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return generate_frames()

@app.route('/')
def index():
    return render_template('index.html', security_mode=security_mode)

@app.route('/video_feed')
def video_feed():
    global security_mode
    if security_mode:
        return Response(sec_cam(), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        return Response(reg_cam(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_security_mode', methods=['POST'])
def toggle_security_mode():
    global security_mode
    security_mode = not security_mode
    return redirect('/')

@app.route('/records')
def records():
    video_files = os.listdir(record_folder)
    return render_template('records.html', video_files=video_files)

@app.route('/play_video/<filename>')
def play_video(filename):
    video_path = os.path.join(record_folder, filename)
    return send_file(video_path, as_attachment=False, mimetype='video/mp4')

if __name__ == '__main__':
    app.run(debug=True)