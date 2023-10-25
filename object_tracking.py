import cv2
import staple_config
from staple import Staple
import operator
import queue
import numpy as np
import camera_control as cc

feature_params = dict(maxCorners=500, qualityLevel=0.03, minDistance=7, blockSize=7)


def get_border(frame, bbox): #, frame_width, frame_height):
    # top = max(bbox[1], 0)
    # bottom = min(bbox[1] + bbox[3], frame_height)
    # left = max(bbox[0], 0)
    # right = min(bbox[0] + bbox[2], frame_width)
    crop_img = frame[bbox[1]:(bbox[1] + bbox[3] + 1), bbox[0]:(bbox[0] + bbox[2] + 1)]
    # if crop_img == []:
    #     return [[0, 0]]
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(crop_img)
    mask[:] = 255
    p = cv2.goodFeaturesToTrack(crop_img, mask=mask, **feature_params) #, useHarrisDetector=True)
    # (cX, cY) = (-1, -1)
    # num = len(p)
    point_list = []
    if p is not None:
        hull = cv2.convexHull(p, returnPoints=True)

        for x, y in np.float32(hull).reshape(-1, 2):
            # cv2.circle(mask, (x, y), 5, (0, 255, 0), -1)
            point_list.append((x + bbox[0], y + bbox[1]))

    if len(point_list) < 3:
        point_list.clear()
        point_list.append([0, 0])

        # if len(point_list) >= 3:
        #     ref_polygon = Polygon(point_list)
        #     cX = int(ref_polygon.centroid.x) + bbox[0]
        #     cY = int(ref_polygon.centroid.y) + bbox[1]
        #
        #     print('update: '+ str(num))
        #     print('update: ' + str(cX) + ' ' + str(cY))

    # cv2.imshow('ROI', mask)
    # return cX, cY
    return point_list


def object_detection(camera, req_buf, viewpos_dict):
    lk_params = dict(winSize=(15, 15), maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    tracker = Staple(config=staple_config.StapleConfig())
    flag_focus = False
    pre_frame = None
    flag_error = False
    es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 2))
    track_list = []
    # pos_q = queue.Queue(maxsize=3)
    dup_list = []
    max_dup_len = 5
    # camera = cv2.VideoCapture(0)
    frame = None
    frame_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH) / 2
    frame_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT) / 2
    frame_size = frame_width * frame_height
    # lm.init_model(frame_width, frame_height)

    while True:
        if flag_error:
            flag_error = False
        else:
            _, frame = camera.read()
            for i in range(0, 4):
                _, frame = camera.read()
            frame = cv2.resize(frame, (int(frame_width), int(frame_height)))

        if flag_focus:
            bbox = tracker.update(frame)
            if len(bbox) == 0:
                flag_focus = False
                flag_error = True
                continue

            bbox = list(map(int, bbox))
            if bbox[2] == 0 or bbox[3] == 0:
                flag_focus = False
                flag_error = True
                continue

            if bbox[0] < 0 or bbox[1] < 0 or bbox[0] + bbox[2] > frame_width or bbox[1] + bbox[3] > frame_height:
                point_list = [[0, 0]]
                # cur_viewpos = cc.get_viewpos()
                cur_viewpos = [viewpos_dict['lat'], viewpos_dict['lon']]
                # cur_viewpos = [0, 0]
                point_list.append(cur_viewpos)
                # req_q.put(point_list)
                req_buf.append(point_list)
                flag_focus = False
                continue

            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            size = bbox[2] * bbox[3]
            # c = (int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2))
            if (size > frame_size / 2) or (size < frame_size / 16):
                flag_focus = False
                flag_error = True
                dup_list.clear()
                continue

            if operator.eq(p1, dup_list[-1][0]) and operator.eq(p2, dup_list[-1][1]):
                dup_list.append((p1, p2))
            if len(dup_list) == max_dup_len + 1:
                flag_focus = False
                flag_error = True
                dup_list.clear()
                continue
            # if pos_q.full():
            #     base_pos = pos_q.get()
            #     if (operator.eq(p1, base_pos[0]) and operator.eq(p2, base_pos[1])) \
            #             or (size > frame_size / 2):
            #         flag_focus = False
            #         flag_error = True
            #         continue

            # cX, cY = get_centroid(frame, bbox)
            # if cX != -1 and cY != -1:
            #     cv2.circle(frame, (cX, cY), 1, (0, 0, 255), 4)
            point_list = get_border(frame, bbox) #, frame_width, frame_height)
            # pos_q.put((p1, p2)) #, (cX,cY)))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2)
            # cur_viewpos = cc.get_viewpos()
            cur_viewpos = [viewpos_dict['lat'], viewpos_dict['lon']]
            # cur_viewpos = [0, 0]
            point_list.append(cur_viewpos)
            # req_q.put(point_list)
            req_buf.append(point_list)
        else: # 如果某个位置一定时间内都没有物体出现，则转回到原点！！！
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(frame_gray)
            mask[:] = 255
            p = cv2.goodFeaturesToTrack(frame_gray, mask=mask, **feature_params) #, useHarrisDetector=True)
            if p is not None:
                for x, y in np.float32(p).reshape(-1, 2):
                    track_list.append([(x, y)])

                if pre_frame is None:
                    pre_frame = frame_gray
                    continue

                pre_points = np.float32([tr[-1] for tr in track_list]).reshape(-1, 1, 2)
                cur_points, _, _ = cv2.calcOpticalFlowPyrLK(pre_frame, frame_gray, pre_points, None, **lk_params)
                pre_points_r, _, _ = cv2.calcOpticalFlowPyrLK(frame_gray, pre_frame, cur_points, None, **lk_params)
                d = abs(pre_points - pre_points_r).reshape(-1, 2).max(-1)
                dis = abs(cur_points - pre_points).reshape(-1, 2).max(-1)
                good = (dis > 1) & (d < 1)

                cur_blur = cv2.GaussianBlur(frame_gray, (21, 21), 0)
                pre_blur = cv2.GaussianBlur(pre_frame, (21, 21), 0)
                diff = cv2.absdiff(pre_blur, cur_blur)
                diff = cv2.threshold(diff, 0, 255, cv2.THRESH_OTSU)[1]
                diff = cv2.dilate(diff, es, iterations=2)
                cnts, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnt_list = []
                for c in cnts:
                    cnt_list.append(cv2.boundingRect(c))
                # cv2.imshow('Diff', diff)

                final_tracks = []
                point_list = []
                for tr, (x, y), good_flag in zip(track_list, cur_points.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    flag_cnt = False
                    for cnt in cnt_list:
                        if int(x) in range(cnt[0], cnt[0]+cnt[2]+1) \
                                and int(y) in range(cnt[1], cnt[1]+cnt[3]+1):
                            flag_cnt = True
                            break

                    if flag_cnt is False:
                        continue

                    tr.append((int(x), int(y)))
                    # if len(tr) > 10:
                    #     del tr[0]
                    final_tracks.append(tr)
                    point_list.append((int(x),int(y)))
                    # cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)
                    # cv2.polylines(frame, [np.int32(tr) for tr in final_tracks], False, (0, 255, 0))

                num = len(point_list)
                points = np.float32(point_list).reshape(-1, 1, 2)
                point_list = []
                hull = cv2.convexHull(points, returnPoints=True)
                if hull is None:
                    continue
                for x, y in np.float32(hull).reshape(-1, 2):
                    point_list.append((x, y))

                if len(point_list) < 3:
                    continue
                # ref_polygon = Polygon(point_list)
                # cX = int(ref_polygon.centroid.x)
                # cY = int(ref_polygon.centroid.y)
                # cv2.circle(frame, (cX, cY), 1, (0, 0, 255), 4)

                final_points = np.float32(point_list).reshape(-1, 2)
                left, top = np.amin(final_points, axis=0)
                right, bottom = np.amax(final_points, axis=0)
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                roi = [left, top, (right - left), (bottom - top)]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                tracker.init(frame, roi)
                # c = (int((left + right) / 2), int((top + bottom) / 2))
                size = (right - left) * (bottom - top)
                if size > frame_size / 2:
                    continue

                # if pos_q.full():
                #     pos_q.get()
                # pos_q.put(((left, top), (right, bottom))) #, (cX, cY)))
                dup_list.append(((left, top), (right, bottom)))

                # cv2.imshow('Detection', frame_gray)
                flag_focus = True
                # cur_viewpos = cc.get_viewpos()
                cur_viewpos = [viewpos_dict['lat'], viewpos_dict['lon']]
                # cur_viewpos = [0, 0]
                point_list.append(cur_viewpos)
                # req_q.put(point_list)
                req_buf.append(point_list)

                # print('init: '+ str(num))
                # print('init: ' + str(cX) + ' ' + str(cY))

            pre_frame = None
            track_list = []

        cv2.imshow('Tracking', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    camera.release()
    cv2.destroyAllWindows()
