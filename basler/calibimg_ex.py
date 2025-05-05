from pypylon import pylon
import cv2
import time
import threading
from datetime import datetime
import os


devices = pylon.TlFactory.GetInstance().EnumerateDevices()
if len(devices) == 0:
    raise pylon.RuntimeException("No camera present.")


def Set_Camera_Properties(camera_id):
    camera_id.Width.SetValue(1440)
    camera_id.Height.SetValue(2000)
    camera_id.OffsetX.SetValue(1920)
    camera_id.OffsetY.SetValue(520)
    camera_id.PixelFormat.SetValue('Mono8')
    camera_id.ExposureTime.SetValue(5000.0)
    print(f'相机ID: {camera_id.GetDeviceInfo().GetSerialNumber()}, 相机分辨率: {camera_id.Width.GetValue()}*{camera_id.Height.GetValue()}, 像素格式: {camera_id.PixelFormat.GetValue()}, 曝光时间: {camera_id.ExposureTime.GetValue()}(us)')


def Format_Converter():
    converter = pylon.ImageFormatConverter()
    converter.OutputPixelFormat = pylon.PixelType_BGR8packed
    converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    return converter


def Save_Image_00(camera_id):
    while camera_id.IsGrabbing():
        with camera_id.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
            if grabResult.GrabSucceeded():
                pylon_img.AttachGrabResultBuffer(grabResult)
                image = Format_Converter().Convert(grabResult)
                cv2_img = image.GetArray()
                cv2.namedWindow('calib_00', cv2.WINDOW_NORMAL)
                cv2.imshow('calib_00', cv2_img)
                k = cv2.waitKey(1)
                if k == ord('s'):
                    file_name = 'calib_00/calib_l_{}.bmp'.format(datetime.now().strftime('%H%M%S'))
                    cv2.imwrite(file_name, cv2_img)
                    # pylon_img.Save(pylon.ImageFileFormat_Bmp, file_name)
                    print(f'save image: {file_name}')
                elif k == ord('q'):
                    break
            pylon_img.Release()
            grabResult.Release()
    camera_id.StopGrabbing()


def Save_Image_01(camera_id):
    while camera_id.IsGrabbing():
        with camera_id.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException) as grabResult:
            if grabResult.GrabSucceeded():
                pylon_img.AttachGrabResultBuffer(grabResult)
                image = Format_Converter().Convert(grabResult)
                cv2_img = image.GetArray()
                cv2.namedWindow('calib_01', cv2.WINDOW_NORMAL)
                cv2.imshow('calib_01', cv2_img)
                k = cv2.waitKey(1)
                if k == ord('s'):
                    file_name = 'calib_01/calib_r_{}.bmp'.format(datetime.now().strftime('%H%M%S'))
                    cv2.imwrite(file_name, cv2_img)
                    # pylon_img.Save(pylon.ImageFileFormat_Bmp, file_name)
                    print(f'save image: {file_name}')
                elif k == ord('q'):
                    break
            pylon_img.Release()
            grabResult.Release()
    camera_id.StopGrabbing()


if __name__ == '__main__':
    dir_path_1 = 'calib_00/'
    dir_path_2 = 'calib_01/'
    if not os.path.exists(dir_path_1) and not os.path.exists(dir_path_2):
        os.mkdir(dir_path_1)
        os.mkdir(dir_path_2)
    else:
        print('已存在文件夹: calib_00/ calib_01/')

    cam_00 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[0]))
    cam_01 = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(devices[1]))

    cam_00.Open()
    cam_01.Open()

    Set_Camera_Properties(cam_00)
    Set_Camera_Properties(cam_01)

    pylon_img = pylon.PylonImage()

    cam_00.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    cam_01.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    t1 = threading.Thread(target=Save_Image_00, args=(cam_00,))
    t2 = threading.Thread(target=Save_Image_01, args=(cam_01,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()