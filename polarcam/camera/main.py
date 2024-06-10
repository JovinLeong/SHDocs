import FLIRPolarCam
import sys
import matplotlib.pyplot as plt
import numpy as np


def main():
    
    polar_cam = FLIRPolarCam.PolarCam()

    fig = plt.figure(1)
    
    #configure sequence
    settings_list = []
    settings_list.append((2448, 2048, 5000, 14)) #exposure time in microseconds, gain is in decibels
    settings_list.append((2448, 2048, 10000, 14))
    settings_list.append((2448, 2048, 15000, 14))
    settings_list.append((2448, 2048, 20000, 14))
    settings_list.append((2448, 2048, 25000, 14))

    polar_cam.configure_image_sequence(settings_list)

    polar_cam.start_acquisition()

    #perform sequence capture. 
    image_result_array = polar_cam.grab_sequence(5)
    
    #display and resize images
    #concat image to a column. 
    np_top_pair = image_result_array[0]

    for i in range(1, 5):
        np_top_pair = np.concatenate((np_top_pair, image_result_array[i]), axis=1)

    polar_cam.stop_acquisition()
    polar_cam.release()

    plt.imshow(np_top_pair, cmap='gray')
    plt.pause(100)
   # plt.clf()

if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)

