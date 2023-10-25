from onvif import ONVIFCamera

cam_ip = '192.168.0.105'
cam_port = 2020
username = ''
passward = ''
input_path = 'rtsp://' + cam_ip + '/stream1'
pan_angle = 336
tilt_angle = 163
min_resolution_height = 720
min_resolution_width = 1280
init_pan_pos = 0
init_tilt_pos = 0
upside_down = False
ptz = None
token = None


def camera_init():
    cam = ONVIFCamera(cam_ip, cam_port, username, passward)
    media = cam.create_media_service()
    global ptz
    ptz = cam.create_ptz_service()
    media_profile = media.GetProfiles()[0]
    global token
    token = media_profile.token

    status = ptz.GetStatus({'ProfileToken': token})
    status.Position.PanTilt.x = init_pan_pos
    status.Position.PanTilt.y = init_tilt_pos
    request = ptz.create_type('AbsoluteMove')
    # request = ptz.create_type('GotoHomePosition')
    request.ProfileToken = token
    request.Position = status.Position

    ptz.Stop({'ProfileToken': token})
    ptz.AbsoluteMove(request)
    # ptz.GotoHomePosition(request)

    config_list = media.GetVideoEncoderConfigurations()
    video_encoder_config = config_list[0]
    options = media.GetVideoEncoderConfigurationOptions({'ProfileToken': token})
    # video_encoder_config.Encoding = 'H264'
    video_encoder_config.Quality = options.QualityRange.Min
    video_encoder_config.Resolution.Height = min_resolution_height
    video_encoder_config.Resolution.Width = min_resolution_width
    # video_encoder_config.RateControl.FrameRateLimit = options.H264.FrameRateRange.Min
    # video_encoder_config.RateControl.EncodingInterval = options.H264.EncodingIntervalRange.Min

    request = media.create_type('SetVideoEncoderConfiguration')
    request.Configuration = video_encoder_config
    request.ForcePersistence = True
    media.SetVideoEncoderConfiguration(request)


def get_viewpos():
    status = ptz.GetStatus({'ProfileToken': token})
    (x, y) = (status.Position.PanTilt.x, status.Position.PanTilt.y)
    if upside_down:
        v_lat = -x * pan_angle / 2
        v_lon = y * tilt_angle / 2
    else:
        v_lat = x * pan_angle / 2
        v_lon = -y * tilt_angle / 2

    if x == 0:
        v_lat = 0.0
    if y == 0:
        v_lon = 0.0

    return v_lat, v_lon


def camera_move(x, y):
    status = ptz.GetStatus({'ProfileToken': token})
    status.Position.PanTilt.x += x
    if status.Position.PanTilt.x < -1:
        status.Position.PanTilt.x = -1
    if status.Position.PanTilt.x > 1:
        status.Position.PanTilt.x = 1
    status.Position.PanTilt.y += y
    if status.Position.PanTilt.y < -1:
        status.Position.PanTilt.y = -1
    if status.Position.PanTilt.y > 1:
        status.Position.PanTilt.y = 1
    request = ptz.create_type('AbsoluteMove')
    request.ProfileToken = token
    request.Position = status.Position

    ptz.Stop({'ProfileToken': token})
    ptz.AbsoluteMove(request)


def camera_control(cmd_q, viewpos_dict, lock):
    # print("Camera control start.")
    camera_init()
    lat, lon = get_viewpos()
    viewpos_dict['lat'] = lat
    viewpos_dict['lon'] = lon
    while True:
        if not cmd_q.empty():
            lat, lon = 0, 0
            lock.acquire()
            pop_msg = cmd_q.get()
            priority = pop_msg[0]
            cur_lat, cur_lon = get_viewpos()
            if priority or (cur_lat == pop_msg[3] and cur_lon == pop_msg[4]):
                lat, lon = pop_msg[1], pop_msg[2]
            while not cmd_q.empty():
                pop_msg = cmd_q.get()
                if cur_lat == pop_msg[3] and cur_lon == pop_msg[4]:
                    lat, lon = pop_msg[1], pop_msg[2]
            lock.release()
            if upside_down:
                camera_move(-lat / pan_angle, lon / tilt_angle)
            else:
                camera_move(lat / pan_angle, -lon / tilt_angle)

            lat, lon = get_viewpos()
            viewpos_dict['lat'] = lat
            viewpos_dict['lon'] = lon
