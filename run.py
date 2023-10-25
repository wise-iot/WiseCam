import cv2
import queue
import object_tracking as ot
import inference_model as im
import camera_control as cc
import threading
import multiprocessing as mp

if __name__ == '__main__':
    try:
        # upside_down = False
        req_buf = mp.Manager().list([])
        viewpos_dict = mp.Manager().dict()
        viewpos_dict['lat'] = 0.0
        viewpos_dict['lon'] = 0.0
        # cmd_q = queue.PriorityQueue()
        cmd_q = mp.Queue()
        lock = mp.Lock()

        cam_ip = '192.168.0.105'
        input_path = 'rtsp://' + cam_ip + '/stream1'
        # input_path = 0
        camera = cv2.VideoCapture(input_path)
        frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
        frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
        width = mp.Value('d', frame_width)
        height = mp.Value('d', frame_height)
        # frame_size = frame_width * frame_height
        # im.init_model(frame_width, frame_height)
        # cc.camera_init()

        # ct = threading.Thread(target=cc.camera_control, args=(cmd_q, upside_down, lock))
        # ct.setDaemon(True)
        # ct.start()

        ip = mp.Process(target=im.run_model, args=(req_buf, cmd_q, lock, width, height))
        cp = mp.Process(target=cc.camera_control, args=(cmd_q, viewpos_dict, lock))
        dt = threading.Thread(target=ot.object_detection, args=(camera, req_buf, viewpos_dict))
        dt.setDaemon(True)
        ip.start()
        cp.start()
        dt.start()
        ip.join()
        cp.join()

    except SystemExit:
        pass
