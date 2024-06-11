import PySpin
import sys
import numpy as np
import matplotlib.pyplot as plt



from FLIRCamHelper import reset_sequencer, configure_sequencer_part_one, configure_sequencer_part_two, set_single_state_from_list_of_tuple, set_cam_exposure_auto, set_cam_gain_auto, set_cam_fps_auto
                        

class PolarCam:

    def __init__(self):
        self.system = PySpin.System.GetInstance()
        # Get current library version
        version = self.system.GetLibraryVersion()
        print('Library version: %d.%d.%d.%d' % (version.major, version.minor, version.type, version.build))

        self.cam_list = self.system.GetCameras()
        num_cameras = self.cam_list.GetSize()

        print('Number of cameras detected: %d' % num_cameras)

        if num_cameras == 0:

            print('Not enough cameras!')

            # Clear camera list before releasing system
            self.cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

            return

        self.processor = PySpin.ImageProcessor()
        
        self.max_shutter_speed_us = 3000000
        self.min_shutter_speed_us = 200
        self.curr_shutter_speed_us = 200

        self.max_gain = 40.0
        self.min_gain = 0.1
        self.curr_gain = 10.0
        
        self.min_fps = 1.0
        self.max_fps = 24.0
        self.curr_fps = 8

        #set the default settings. 
        for i, cam in enumerate(self.cam_list):
            self.configure_default_settings(cam)


    def __del__(self):
        self.release()

    def release(self):

        try:
            for i, cam in enumerate(self.cam_list):
                cam.DeInit()
                del cam

            # Clear camera list before releasing system
            self.cam_list.Clear()

            # Release system instance
            self.system.ReleaseInstance()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False

    def configure_default_settings(self, cam):
        
        cam.Init()
        #configure stream
        self.configure_stream_settings(cam)
        self.configure_exposure_control(cam) #extract minimum and maximum exposure settings
        self.configure_acquisition_control(cam)

        self.configure_gain_control(cam)
        self.configure_fps_control(cam)


    def get_curr_exposure_value(self) -> float:
        for i, cam in enumerate(self.cam_list):
            if cam.ExposureTime.GetAccessMode() == PySpin.RW or cam.ExposureTime.GetAccessMode() == PySpin.RO:
                return cam.ExposureTime.GetValue()

        return 0.0
    
    def set_exposure_auto(self, on_or_off):

        if on_or_off :
             for i, cam in enumerate(self.cam_list):
                nodemap = cam.GetNodeMap()
                set_cam_exposure_auto(nodemap, on_or_off)


    def get_curr_gain_value(self):
        for i, cam in enumerate(self.cam_list):
            if cam.Gain.GetAccessMode() == PySpin.RW or cam.Gain.GetAccessMode() == PySpin.RO:
                return cam.Gain.GetValue()

        return 0
    
    def set_gain_auto(self, on_or_off):

        if on_or_off :
             for i, cam in enumerate(self.cam_list):
                nodemap = cam.GetNodeMap()
                set_cam_gain_auto(nodemap, on_or_off)

    def configure_fps_control(self, cam):
        """Get current, min, max fps of camera
        
        """
        if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW or cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RO:
            self.curr_fps = cam.AcquisitionFrameRate.GetValue()
            self.min_fps = cam.AcquisitionFrameRate.GetMin()
            self.max_fps = cam.AcquisitionFrameRate.GetMax()
            print('Updated Min/Max FPS : {0}, {1}, {2}'.format(self.min_fps, self.max_fps, self.curr_fps))
    
    def get_fps_value(self) -> float:
        """Get FPS value

        Returns:
            float: _description_
        """
        for i, cam in enumerate(self.cam_list):
            if cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RW or cam.AcquisitionFrameRate.GetAccessMode() == PySpin.RO:
                return cam.AcquisitionFrameRate.GetValue()

        return 0.0
    
    def set_fps_value_from_step(self, fps_slider_val):

        for i, cam in enumerate(self.cam_list):
            try:
                print('*** CONFIGURING FPS ***\n')

                if cam.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
                    print('Unable to disable automatic FPS. Aborting...')
                    return False
                
                cam.AcquisitionFrameRate.SetValue(PySpin.GainAuto_Off)
                print('Automatic FPS disabled...')

                if cam.AcquisitionFrameRate.GetAccessMode() != PySpin.RW:
                    print('Unable to set FPS value. Aborting...')
                    return False

                fps_to_set = fps_slider_val * (self.max_fps - self.min_fps)/100

                cam.AcquisitionFrameRate.SetValue(fps_to_set)
                print('FPS set to %s us...\n' % fps_to_set)

                return True
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False      


    def set_fps_auto(self, on_or_off):
        if on_or_off :
             for i, cam in enumerate(self.cam_list):
                nodemap = cam.GetNodeMap()
                set_cam_fps_auto(nodemap, on_or_off)

    def configure_gain_control(self, cam):
        
        if cam.Gain.GetAccessMode() == PySpin.RW or cam.Gain.GetAccessMode() == PySpin.RO:
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit being used in GetNextImage

            #current exposure time
            self.min_gain = cam.Gain.GetMin()
            self.max_gain = cam.Gain.GetMax()
            self.curr_gain = cam.Gain.GetValue()
            print('Updated Min/Max Gain : {0}, {1}, {2}'.format(self.min_gain, self.max_gain, self.curr_gain))
    
    def set_gain_value_from_step(self, gain_slider_val):

        for i, cam in enumerate(self.cam_list):
            try:
                print('*** CONFIGURING GAIN ***\n')

                if cam.GainAuto.GetAccessMode() != PySpin.RW:
                    print('Unable to disable automatic gain. Aborting...')
                    return False
                
                cam.GainAuto.SetValue(PySpin.GainAuto_Off)
                print('Automatic Gain disabled...')

                if cam.Gain.GetAccessMode() != PySpin.RW:
                    print('Unable to set gain value. Aborting...')
                    return False

                gain_to_set = gain_slider_val * (self.max_gain - self.min_gain)/100

                cam.Gain.SetValue(gain_to_set)
                print('Gain set to %s us...\n' % gain_to_set)

                return True
            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False      


    def configure_exposure_control(self, cam):

        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        node_exposure_lighting_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AutoExposureLightingMode'))
        if not PySpin.IsReadable(node_exposure_lighting_mode) or not PySpin.IsWritable(node_exposure_lighting_mode):
            print('\nUnable to set Exposure Lighting Mode. Aborting...\n')
            return False

        exposure_mode_front_light = node_exposure_lighting_mode.GetEntryByName('Frontlight')
        if not PySpin.IsReadable(exposure_mode_front_light):
            print('\nUnable to set Exposure Front Light. Aborting...\n')
            return False

        #set to front lighting
        lighting_front_light = exposure_mode_front_light.GetValue()
        node_exposure_lighting_mode.SetIntValue(lighting_front_light)

        if cam.ExposureTime.GetAccessMode() == PySpin.RW or cam.ExposureTime.GetAccessMode() == PySpin.RO:
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit being used in GetNextImage

            #current exposure time 
            self.min_shutter_speed_us = cam.ExposureTime.GetMin()
            self.max_shutter_speed_us = cam.ExposureTime.GetMax()
            self.curr_shutter_speed_us = cam.ExposureTime.GetValue()
            print('Updated Min/Max Shutter Speed: {0}, {1}, {2}'.format(self.min_shutter_speed_us, self.max_shutter_speed_us, self.curr_shutter_speed_us))

    def set_exposure_time_from_step(self, shutter_slider_val):
        
        #get first cam
        for i, cam in enumerate(self.cam_list):
            try:
                print('*** CONFIGURING EXPOSURE ***\n')

                if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                    print('Unable to disable automatic exposure. Aborting...')
                    return False

                cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
                print('Automatic exposure disabled...')

                if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                    print('Unable to set exposure time. Aborting...')
                    return False

                # Ensure desired exposure time does not exceed the maximum
                exposure_time_max = min(self.max_shutter_speed_us, 500000)  

                if (exposure_time_max < self.min_shutter_speed_us):
                    exposure_time_max = self.min_shutter_speed_us

                exposure_time_to_set = shutter_slider_val * (exposure_time_max - self.min_shutter_speed_us)/100 + self.min_shutter_speed_us

                cam.ExposureTime.SetValue(exposure_time_to_set)
                print('Shutter time set to %s us...\n' % exposure_time_to_set)

                return True

            except PySpin.SpinnakerException as ex:
                print('Error: %s' % ex)
                return False  

    def set_exposure_time(self, cam, shutter_us):
        """Sets the exposure time, shutter speed in microseconds. 

        Args:
            cam (_type_): _description_
            shutter_us (_type_): _description_

        Returns:
            _type_: _description_
        """
        # Likely need to set upper and lower limit of exposure or modify actual exposure. 
        try:
            print('*** CONFIGURING EXPOSURE ***\n')

            if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                print('Unable to disable automatic exposure. Aborting...')
                return False

            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Off)
            print('Automatic exposure disabled...')

            # Set exposure time manually; exposure time recorded in microseconds
            #
            # *** NOTES ***
            # Notice that the node is checked for availability and writability
            # prior to the setting of the node. In QuickSpin, availability and
            # writability are ensured by checking the access mode.
            #
            # Further, it is ensured that the desired exposure time does not exceed
            # the maximum. Exposure time is counted in microseconds - this can be
            # found out either by retrieving the unit with the GetUnit() method or
            # by checking SpinView.

            if cam.ExposureTime.GetAccessMode() != PySpin.RW:
                print('Unable to set exposure time. Aborting...')
                return False

            # Ensure desired exposure time does not exceed the maximum
            exposure_time_to_set = shutter_us
            exposure_time_to_set = min(cam.ExposureTime.GetMax(), exposure_time_to_set)
            cam.ExposureTime.SetValue(exposure_time_to_set)
            print('Shutter time set to %s us...\n' % exposure_time_to_set)

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

    def reset_exposure(cam):
        """
        This function returns the camera to a normal state by re-enabling automatic exposure.

        :param cam: Camera to reset exposure on.
        :type cam: CameraPtr
        :return: True if successful, False otherwise.
        :rtype: bool
        """
        try:
            result = True

            # Turn automatic exposure back on
            #
            # *** NOTES ***
            # Automatic exposure is turned on in order to return the camera to its
            # default state.

            if cam.ExposureAuto.GetAccessMode() != PySpin.RW:
                print('Unable to enable automatic exposure (node retrieval). Non-fatal error...')
                return False

            cam.ExposureAuto.SetValue(PySpin.ExposureAuto_Continuous)

            print('Automatic exposure enabled...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            result = False

        return result

    def configure_camera_to_polarized8_format(self):
        
        for i, cam in enumerate(self.cam_list):

            try:
                # Turn sequencer mode back off
                #
                # *** NOTES ***
                # The sequencer is turned off in order to return the camera to its default state.
                nodemap = cam.GetNodeMap()
                node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
                if not PySpin.IsReadable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
                    print('Unable to access SequencerMode state')
                    continue

                sequencer_mode_off = node_sequencer_mode.GetEntryByName('Off')
                if not PySpin.IsReadable(sequencer_mode_off):
                    print('Unable to get SequencerMode off')
                    continue

                node_sequencer_mode.SetIntValue(sequencer_mode_off.GetValue())

                print('Turning off sequencer mode...')

                print('*** Setting IMAGE Width, Height, Pixel Format ***\n')
                #Set camera to get image in polarized8 format. 
                if cam.PixelFormat.GetAccessMode() == PySpin.RW:
                    cam.PixelFormat.SetValue(PySpin.PixelFormat_Polarized8)
                    print('Pixel format set to %s...' % cam.PixelFormat.GetCurrentEntry().GetSymbolic())
                else:
                    print('Pixel format not available...')

            except PySpin.SpinnakerException as ex:
                print('Error: {} reset_sequencer '.format(ex))
        

    def configure_acquisition_control(self, cam):
        
        #set the width and height. 
        # Retrieve GenICam nodemap
        nodemap = cam.GetNodeMap()

        print('*** Setting IMAGE Width, Height, Pixel Format ***\n')
        #Set camera to get image in polarized8 format. 
        if cam.PixelFormat.GetAccessMode() == PySpin.RW:
            cam.PixelFormat.SetValue(PySpin.PixelFormat_Polarized8)
            print('Pixel format set to %s...' % cam.PixelFormat.GetCurrentEntry().GetSymbolic())
        else:
            print('Pixel format not available...')
            result = False

         # Set maximum width
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):

            width_to_set = node_width.GetMax()
            node_width.SetValue(width_to_set)
            print('Width set to %i...' % node_width.GetValue())

        else:
            print('Width not readable or writable...')

        # Set maximum height
        #
        # *** NOTES ***
        # A maximum is retrieved with the method GetMax(). A node's minimum and
        # maximum should always be a multiple of its increment.
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if  PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):

            height_to_set = node_height.GetMax()
            node_height.SetValue(height_to_set)
            print('Height set to %i...' % node_height.GetValue())

        else:
            print('Height not readable or writable...')

        print('*** Setting IMAGE ACQUISITION Mode ***\n')

        try:
            node_acquisition_mode = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionMode'))
            if not PySpin.IsReadable(node_acquisition_mode) or not PySpin.IsWritable(node_acquisition_mode):
                print('Unable to set acquisition mode to continuous (enum retrieval). Aborting...')
                return False

            # Retrieve entry node from enumeration node
            node_acquisition_mode_continuous = node_acquisition_mode.GetEntryByName('Continuous')
            if not PySpin.IsReadable(node_acquisition_mode_continuous):
                print('Unable to set acquisition mode to continuous (entry retrieval). Aborting...')
                return False
            
            # Retrieve integer value from entry node
            acquisition_mode_continuous = node_acquisition_mode_continuous.GetValue()

            # Set integer value from entry node as new value of enumeration node
            node_acquisition_mode.SetIntValue(acquisition_mode_continuous)

            print('Acquisition mode set to continuous...')

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)
            return False
    
    def configure_stream_settings(self, cam):
        """Configure stream settings. Default: Set Stream to Newest Only

        Args:
            cam (_type_): _description_

        Returns:
            _type_: _description_
        """
        sNodemap = cam.GetTLStreamNodeMap()

        # Change bufferhandling mode to NewestOnly
        node_bufferhandling_mode = PySpin.CEnumerationPtr(sNodemap.GetNode('StreamBufferHandlingMode'))
        if not PySpin.IsReadable(node_bufferhandling_mode) or not PySpin.IsWritable(node_bufferhandling_mode):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve entry node from enumeration node
        node_newestonly = node_bufferhandling_mode.GetEntryByName('NewestOnly')
        if not PySpin.IsReadable(node_newestonly):
            print('Unable to set stream buffer handling mode.. Aborting...')
            return False

        # Retrieve integer value from entry node
        node_newestonly_mode = node_newestonly.GetValue()

        # Set integer value from entry node as new value of enumeration node
        node_bufferhandling_mode.SetIntValue(node_newestonly_mode)

    def start_acquisition(self):
        for i, cam in enumerate(self.cam_list):
            self.start_acquisition_cam(cam)

    def start_acquisition_cam(self, cam):
        cam.BeginAcquisition()

    def stop_acquisition(self):
        for i, cam in enumerate(self.cam_list):
            self.stop_acquisition_cam(cam)

    def stop_acquisition_cam(self, cam):
        try:
            # Initialize camera
            cam.EndAcquisition()
           
        except PySpin.SpinnakerException as ex:
            print('Error: %s stop_acquisition_cam' % ex)

    @staticmethod
    def append_images_to_panel(image_polarized_i0, image_polarized_i45, image_polarized_i90, image_polarized_i135, image_s0_norm, image_deglared):
        np_top_pair = np.concatenate((image_polarized_i0, image_polarized_i45), axis=1)
        np_bottom_pair = np.concatenate((image_polarized_i90, image_polarized_i135), axis=1)
        
        np_top_row = np.concatenate((np_top_pair, image_s0_norm), axis=1)
        np_bot_row = np.concatenate((np_bottom_pair, image_deglared), axis=1)
        image_data = np.concatenate((np_top_row, np_bot_row), axis=0)

        return image_data
        
    def grab_all_polarized_image(self, image_result):
        """Extract the polarization images from the raw polarized 8 image. 

        Args:
            image_result (_type_): returns 6 images of numpy array i0, i45, i90, i135. s0, deglared
        """
        image_polarized_i0 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, PySpin.SPINNAKER_POLARIZATION_QUADRANT_I0).GetNDArray()
        image_polarized_i45 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, PySpin.SPINNAKER_POLARIZATION_QUADRANT_I45).GetNDArray()
        image_polarized_i90= PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, PySpin.SPINNAKER_POLARIZATION_QUADRANT_I90).GetNDArray()
        image_polarized_i135 = PySpin.ImageUtilityPolarization.ExtractPolarQuadrant(image_result, PySpin.SPINNAKER_POLARIZATION_QUADRANT_I135).GetNDArray()

        image_s0 = PySpin.ImageUtilityPolarization.CreateStokesS0(image_result)
        image_s0_norm = PySpin.ImageUtility.CreateNormalized(image_s0, PySpin.PixelFormat_Mono8, PySpin.SPINNAKER_SOURCE_DATA_RANGE_ABSOLUTE_DATA_RANGE).GetNDArray()
        image_deglared = PySpin.ImageUtilityPolarization.CreateGlareReduced(image_result).GetNDArray()

        return image_polarized_i0, image_polarized_i45, image_polarized_i90, image_polarized_i135, image_s0_norm, image_deglared

    def grab_image(self):
        for i, cam in enumerate(self.cam_list):
            return self.grab_image_cam(cam)
        
        return None
        
    def grab_image_cam(self, cam):
        
        timeout = 0
        if cam.ExposureTime.GetAccessMode() == PySpin.RW or cam.ExposureTime.GetAccessMode() == PySpin.RO:
            # The exposure time is retrieved in µs so it needs to be converted to ms to keep consistency with the unit being used in GetNextImage
            timeout = (int)(cam.ExposureTime.GetValue() / 1000 + 1000)
        else:
            print ('Unable to get exposure time. Aborting...')
            return None
        
        #timeout is calculated based on current exposure
        image_result = cam.GetNextImage(timeout)
        if image_result.IsIncomplete():
            print('Image incomplete with image status %d ...' % image_result.GetImageStatus())
            return None
        
        image_copy = PySpin.Image.Create()
        image_copy.DeepCopy(image_result)
        return image_copy

    def configure_image_sequence(self, settings_list):
        
        result = True
        for i, cam in enumerate(self.cam_list):
            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            #Set camera to get image in polarized8 format. 
            if cam.PixelFormat.GetAccessMode() == PySpin.RW:
                cam.PixelFormat.SetValue(PySpin.PixelFormat_Polarized8)
                print('Pixel format set to %s...' % cam.PixelFormat.GetCurrentEntry().GetSymbolic())
            else:
                print('Pixel format not available...')
            
            # Configure sequencer to be ready to set sequences
            result &= configure_sequencer_part_one(nodemap)
            if not result:
                return result

            #configure the sequence settings
            set_single_state_from_list_of_tuple(nodemap, settings_list)

            # Configure sequencer to acquire images
            result &= configure_sequencer_part_two(nodemap)
            if not result:
                return result

        return True

    def grab_sequence(self, num_images):
        
        for i, cam in enumerate(self.cam_list):
            img_array = self.grab_image_sequence(cam, num_images)
            return img_array

        
    def grab_image_sequence(self, cam, num_images):
        
        img_out = []
        for i in range(num_images):
            img_copy = self.grab_image_cam(cam)
            if img_copy != None:
                print('Grabbed image sequence: ', i)
                img_out.append(img_copy.GetNDArray())

        return img_out

    def reset_sequencer(self):
        for i, cam in enumerate(self.cam_list):
            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            reset_sequencer(nodemap)
            self.configure_acquisition_control(cam)


    def run_single_camera(self, cam):

        try:
            nodemap_tldevice = cam.GetTLDeviceNodeMap()

            # Initialize camera
            cam.Init()
            
            # Retrieve GenICam nodemap
            nodemap = cam.GetNodeMap()

            cam.DeInit()

        except PySpin.SpinnakerException as ex:
            print('Error: %s' % ex)


def handle_close(evt):
    """
    This function will close the GUI when close event happens.

    :param evt: Event that occurs when the figure closes.
    :type evt: Event
    """

    global start_capturing
    start_capturing = False

def main():

    polar_cam = PolarCam()
    polar_cam.start_acquisition()

    fig = plt.figure(1)
    fig.canvas.mpl_connect('close_event', handle_close)

    counter = 0
    global start_capturing
    start_capturing = True

    #put a while loop. Grab the frames. 
    while start_capturing:

        print("Counter: ", counter)
        #grab image
        image_result = polar_cam.grab_image()

        if image_result == None:
            print("[Error] Unable to capture image")
            break
        
        #extract polarized image
        image_polarized_i0, image_polarized_i45, image_polarized_i90, image_polarized_i135, image_s0_norm, image_deglared = polar_cam.grab_all_polarized_image(image_result)
        image_display = PolarCam.append_images_to_panel(image_polarized_i0, image_polarized_i45, image_polarized_i90, image_polarized_i135, image_s0_norm, image_deglared)
        
                    # If user presses enter, close the program
        
        plt.imshow(image_display, cmap='gray')
        plt.pause(0.001)
        plt.clf()

        counter+=1


    polar_cam.stop_acquisition()


if __name__ == '__main__':
    if main():
        sys.exit(0)
    else:
        sys.exit(1)
