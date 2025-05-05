import cv2
from pypylon import pylon
import threading
from datetime import datetime
import os


devices = pylon.TlFactory.GetInstance().EnumerateDevices()
if len(devices) == 0:
    raise pylon.RuntimeException("No camera present.")


def Set_Camera_Properties(camera_id):
    camera_id.Width.SetValue(1920)
    camera_id.Height.SetValue(1080)
    camera_id.OffsetX.SetValue(1680)
    camera_id.OffsetY.SetValue(980)
    camera_id.PixelFormat.SetValue('RGB8')
    camera_id.ExposureTime.SetValue(5000.0)


def Trigger_On(camera_id):
    camera_id.LineSelector = "Line1"
    camera_id.LineMode = "Input"
    camera_id.TriggerSelector = "FrameStart"
    camera_id.TriggerMode = "On"
    camera_id.TriggerSource = "Line1"
    camera_id.TriggerActivation = "RisingEdge"
    # camera_id.TriggerDelay.SetValue(10000.0)


def Format_Converter():
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def Trigger_Save_00(camera_id):
    print('waiting for trigger signal / 5000(ms)......')
    while camera_id.IsGrabbing():
        with camera_id.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
            if grabResult.GrabSucceeded():
                img_num = grabResult.ImageNumber
                frame_counts_0 = img_num
                # print('cam_00_' + str(img_num))
                # Pylon 格式
                pylon_img.AttachGrabResultBuffer(grabResult)
                # OpenCV 格式
                image = Format_Converter().Convert(grabResult)
                cv2_img = image.GetArray()

                file_name = 'cam_00/cam_l_{}_({:0>3d}).bmp'.format(datetime.now().strftime('%H%M%S.%f'), img_num)
                time_in_time_list_0.append(datetime.now())

                print(file_name, frame_counts_0)
                cv2.imwrite(file_name, cv2_img)
                # pylon_img.Save(pylon.ImageFileFormat_Bmp, file_name)

                if frame_counts_0 >= frames_to_grab:
                    print(f"cam_00 have acquired {frames_to_grab} frames")
                    break
            pylon_img.Release()
        grabResult.Release()
    camera_id.StopGrabbing()
    camera_id.Close()


def Trigger_Save_01(camera_id):
    print('waiting for trigger signal / 5000(ms)......')
    while camera_id.IsGrabbing():
        with camera_id.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
            if grabResult.GrabSucceeded():
                img_num = grabResult.ImageNumber
                frame_counts_1 = img_num
                # print('cam_01_' + str(img_num))
                # Pylon 格式
                pylon_img.AttachGrabResultBuffer(grabResult)
                # OpenCV 格式
                image = Format_Converter().Convert(grabResult)
                cv2_img = image.GetArray()

                file_name = 'cam_01/cam_r_{}_({:0>3d}).bmp'.format(datetime.now().strftime('%H%M%S.%f'), img_num)
                time_in_time_list_1.append(datetime.now())

                print(file_name, frame_counts_1)
                cv2.imwrite(file_name, cv2_img)
                # pylon_img.Save(pylon.ImageFileFormat_Bmp, file_name)

                if frame_counts_1 >= frames_to_grab:
                    print(f"cam_01 have acquired {frames_to_grab} frames")
                    break
            pylon_img.Release()
        grabResult.Release()
    camera_id.StopGrabbing()
    camera_id.Close()


if __name__ == '__main__':
    # 设置保存路径
    dir_path_1 = 'cam_00/'
    dir_path_2 = 'cam_01/'
    if not os.path.exists(dir_path_1) and not os.path.exists(dir_path_2):
        os.mkdir(dir_path_1)
        os.mkdir(dir_path_2)
    else:
        print('已存在文件夹: cam_00/ cam_01/')

    time_in_time_list_0 = []
    time_in_time_list_1 = []

    cam_00 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
    cam_01 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[1]))

    cam_00.Open()
    cam_01.Open()

    Set_Camera_Properties(cam_00)
    Set_Camera_Properties(cam_01)

    Trigger_On(cam_00)
    Trigger_On(cam_01)

    print(f'ID: {cam_00.GetDeviceInfo().GetSerialNumber()}, W*H: {cam_00.Width.GetValue()}*{cam_00.Height.GetValue()}, exp_time: {cam_00.ExposureTime.GetValue()}(us), pix_format: {cam_00.PixelFormat.GetValue()}, HW_trigger: {cam_00.TriggerMode.GetValue()}, trigger_delay: {cam_00.TriggerDelay.GetValue()}(us)')
    print(f'ID: {cam_01.GetDeviceInfo().GetSerialNumber()}, W*H: {cam_01.Width.GetValue()}*{cam_01.Height.GetValue()}, exp_time: {cam_01.ExposureTime.GetValue()}(us), pix_format: {cam_01.PixelFormat.GetValue()}, HW_trigger: {cam_01.TriggerMode.GetValue()}, trigger_delay: {cam_01.TriggerDelay.GetValue()}(us)')

    pylon_img = pylon.PylonImage()

    cam_00.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    cam_01.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    frames_to_grab = 30
    frame_counts_0 = 0
    frame_counts_1 = 0

    pylon_img = pylon.PylonImage()

    t1 = threading.Thread(target=Trigger_Save_00, args=(cam_00,))
    t2 = threading.Thread(target=Trigger_Save_01, args=(cam_01,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    # print('time_in_time_list_0: ', time_in_time_list_0)
    # print('time_in_time_list_1: ', time_in_time_list_1)
    # 左右相机 前后帧 采集时间差
    dif_list_0 = [(time_in_time_list_0[i+1] - time_in_time_list_0[i]).microseconds for i in range(len(time_in_time_list_0)-1)]
    dif_list_1 = [(time_in_time_list_1[i+1] - time_in_time_list_1[i]).microseconds for i in range(len(time_in_time_list_1)-1)]
    print('dif_list_0(us):', dif_list_0)
    print('dif_list_1(us):', dif_list_1)
    # 左右相机的相同时间帧的时间差 (大减小)
    dif_two_list = [(time_in_time_list_1[i] - time_in_time_list_0[i]).microseconds for i in range(frames_to_grab)]
    # print('dif_2_original', dif_two_list)
    for i in range(len(dif_two_list)):
        if dif_two_list[i] >= 500000:
            dif_two_list[i] = 1000000 - dif_two_list[i]

    print('dif_2_list(us):', dif_two_list)


'''
### 修改之前可用版本

def save_image(camera_id):
    print('等待触发信号/5000(ms)......')
    while camera_id.IsGrabbing():
        with camera_id.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
            if grabResult.GrabSucceeded():
                img_num = grabResult.ImageNumber
                cam_id = grabResult.GetCameraContext()
                frame_counts[cam_id] = img_num
                print('cam#1_' + str(img_num))
                pylon_img.AttachGrabResultBuffer(grabResult)
                grabResult.Release()
                pylon_img.Save(pylon.ImageFileFormat_Jpeg, f'cam_{cam_id:02}/' + f'pylon_{img_num}.jpg')
                pylon_img.Release()
                if min(frame_counts) >= frames_to_grab:
                    print(f"all cameras have acquired {frames_to_grab} frames")
                    break


cam_array = pylon.InstantCameraArray(len(devices))
for idx, camera in enumerate(cam_array):
    camera.SetCameraContext(idx)
    camera.Attach(pylon.TlFactory.GetInstance().CreateDevice(devices[idx]))
    camera.Open()
    Set_Camera_Properties(camera)
    Trigger_On(camera)
    print(f'相机ID: {camera.GetDeviceInfo().GetSerialNumber()}, 相机分辨率: {camera.Width.GetValue()}*{camera.Height.GetValue()}, 曝光时间: {camera.ExposureTime.GetValue()}(us), 像素格式: {camera.PixelFormat.GetValue()}, gain: {camera.Gain.GetValue()}, 硬触发: {camera.TriggerMode.GetValue()}, 触发延时时间: {camera.TriggerDelay.GetValue()}(us)')


frames_to_grab = 20
frame_counts = [0]*len(devices)

pylon_img = pylon.PylonImage()

cam_array.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

t1 = threading.Thread(target=save_image, args=(cam_array,))
t1.start()
t1.join()

cam_array.StopGrabbing()
cam_array.Close()
'''