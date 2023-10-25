# -*- coding: utf-8 -*-

import numpy as np
import math
import itertools
import queue
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
import tensorflow.compat.v1 as tf
from a2c_lstm import A2CLSTM
from shapely.geometry import Polygon
import operator
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


gamma = 0.9
alpha = 1
beta = 1
# frame_size = []
grid_angle = 15
w_view_angle = 117
h_view_angle = 0
# width = 0
# height = 0
model = None
action_space = None


def gen_actions(total, ser_num):
    tmp = list(itertools.product(range(total), repeat=ser_num))
    result = []
    for value in tmp:
        result.append(list(value))
    result = np.array(result)
    return result


def get_pos(point_list, view_pos, width, height):
    # [width, height] = frame_size
    # h_view_angle = w_view_angle * height / width
    ref_polygon = Polygon(point_list)
    cX = ref_polygon.centroid.x
    cY = ref_polygon.centroid.y
    max_x = point_list[np.argmax([abs(p[0] - width / 2) for p in point_list])][0]
    max_y = point_list[np.argmax([abs(p[1] - height / 2) for p in point_list])][1]
    c_lat = view_pos[0] - (cX - width / 2) * w_view_angle / width
    m_lat = view_pos[0] - (max_x - width / 2) * w_view_angle / width
    c_lon = view_pos[1] - (cY - height / 2) * h_view_angle / height
    m_lon = view_pos[1] - (max_y - height / 2) * h_view_angle / height
    return [c_lat, c_lon, m_lat, m_lon]


def cal__reward(pre_pos, cur_pos, action, i_lost):
    # [width, height] = frame_size
    # h_view_angle = w_view_angle * height / width
    r_pos, r_ori = 0, 0
    r_lost = -10
    if i_lost == 0:
        cur_m_lat, cur_m_lon = cur_pos[2], cur_pos[3]
        r_pos = (w_view_angle / 2 - abs(cur_m_lat)) / (w_view_angle / 2) \
                + (h_view_angle / 2 - abs(cur_m_lon)) / (h_view_angle / 2) - 1

        cur_c_lat, cur_c_lon = cur_pos[0], cur_pos[1]
        pre_c_lat, pre_c_lon = pre_pos[0], pre_pos[1]
        vec_vel = (cur_c_lat - pre_c_lat, cur_c_lon - pre_c_lon)
        vec_pos = (-cur_c_lat, -cur_c_lon)
        r_ori = np.dot(vec_vel, vec_pos) / (np.linalg.norm(vec_vel) * np.linalg.norm(vec_pos))
        # r_ori = (vec_vel[0] * vec_pos[0] + vec_vel[1] * vec_pos[1]) / \
        #         math.sqrt((vec_vel[0] ** 2 + vec_vel[1] ** 2) * (vec_pos[0] ** 2 + vec_pos[1] ** 2))

    max_rot = action_space[-1]
    r_rot = 3 * (1 - 2 * abs(action[0]) / max_rot[0]) + 5 * (1 - 2 * abs(action[1]) / max_rot[1])
    reward = i_lost * r_lost + (1 - i_lost) * (r_pos + alpha * r_ori) + beta * r_rot
    return reward


def init_model(width, height):
    global h_view_angle
    h_view_angle = w_view_angle * height / width

    ser_cat = ['Pan_Rot', 'Tilt_Rot']
    features = ['Cent_Lat', 'Cent_Lon', 'Marg_Lat', 'Marg_Lon']
    lr_a = 0.01
    lr_c = 0.02
    entropy_beta = 0.001
    n_scope = 2

    global action_space
    action_space = (gen_actions(n_scope * 2 + 1, len(ser_cat)) - n_scope) * grid_angle
    n_actions = len(action_space)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    global model
    model = A2CLSTM(sess, n_actions, len(features), lr_a, lr_c, entropy_beta)


def run_model(req_buf, cmd_q, lock, w, h):
    # global width
    width = w.value
    # global height
    height = h.value
    init_model(width, height)
    epoch = 0
    action_ser = -1
    mview_pos = [0, 0]
    cur_action = [[0, 0]]
    stat_buf = []
    buf_len = 8

    while True:
        if len(req_buf) != 0:
            lst = req_buf.pop(0)
            view_pos = lst[-1]
            point_list = lst[:-1]

            i_lost = 0
            priority = False
            v_lat, v_lon = view_pos[0], view_pos[1]
            if len(point_list) == 1 and operator.eq(point_list[0], [0, 0]):
                i_lost = 1
                priority = True
                if len(stat_buf) == 0:
                    continue
                sta = stat_buf[-1]
                action = [round((sta[0] - v_lat) / grid_angle) * grid_angle,
                          round((sta[1] - v_lon) / grid_angle) * grid_angle]
                stat_buf.append(stat_buf[-1])
                print('Obj lost')
            else:
                stat_buf.append(get_pos(point_list, view_pos, width, height))
                print('Obj tracked ' + str(stat_buf[-1][0]) + ' ' + str(stat_buf[-1][1]) + ' '
                    + str(stat_buf[-1][2]) + ' ' + str(stat_buf[-1][3]))

            if len(stat_buf) < buf_len:
                # if i_lost == 1:
                #     cur_action[0] = action
                #     stat_buf.clear()
                continue

            # s = [[st[0] - v_lat, st[1] - v_lon, st[2] - v_lat, st[3] - v_lon] for st in stat_buf[:buf_len]]
            s = [[st[0] - mview_pos[0], st[1] - mview_pos[1], st[2] - mview_pos[0], st[3] - mview_pos[1]] for st in stat_buf[:buf_len]]
            if len(stat_buf) > buf_len:
                epoch += 1
                # nv_lat = v_lat + cur_action[0][0]
                # nv_lon = v_lon + cur_action[0][1]
                mview_pos[0] = mview_pos[0] + cur_action[0][0]
                mview_pos[1] = mview_pos[1] + cur_action[0][1]
                s_ = [[st[0] - mview_pos[0], st[1] - mview_pos[1], st[2] - mview_pos[0], st[3] - mview_pos[1]] for st in stat_buf[1:]]
                v_s_ = model.target_v(s_)
                reward = cal__reward(s_[-2], s_[-1], cur_action[0], i_lost)
                td_target = reward + gamma * v_s_
                # global action_ser
                feed_dict = {
                    model.s: s,
                    model.a: np.vstack([action_ser]),
                    model.td_target: np.vstack([td_target])
                }
                model.learn(feed_dict)
                stat_buf.pop(0)

            if i_lost == 1:
                cur_action[0] = action
                stat_buf.clear()
                new_v_lat = v_lat + action[0]
                new_v_lon = v_lon + action[1]
                lock.acquire()
                while not cmd_q.empty():
                    cmd_q.get()
                cmd_q.put([priority, new_v_lat, new_v_lon, v_lat, v_lon])
                lock.release()
                # print('Cam rotated (lost)' + str(v_lat) + ' ' + str(v_lon) + ' ' + str(cur_action[0][0]) + ' '
                #       + str(cur_action[0][1]) + ' ' + str(new_v_lat) + ' ' + str(new_v_lon))
            else:
                # if len(s) == 0:
                #     continue
                if epoch > 500:
                    action_ser = model.choose_action(s)
                    if action_ser != -1:
                        cur_action[0] = action_space[action_ser]
                        new_v_lat = v_lat + cur_action[0][0]
                        new_v_lon = v_lon + cur_action[0][1]
                        cmd_q.put([priority, new_v_lat, new_v_lon, v_lat, v_lon])
                        print('Cam rotated (learning)' + str(v_lat) + ' ' + str(v_lon) + ' ' + str(cur_action[0][0]) + ' '
                              + str(cur_action[0][1]) + ' ' + str(new_v_lat) + ' ' + str(new_v_lon))


