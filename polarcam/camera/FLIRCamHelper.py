import PySpin

def print_retrieve_node_failure(node, name):
    """"
    This function handles the error prints when a node or entry is unavailable or
    not readable on the connected camera.

    :param node: Node type. "Node' or 'Entry'
    :param name: Node name.
    :type node: String
    :type name: String
    :rtype: None
    """
    print('Unable to get {} ({} {} retrieval failed.)'.format(node, name, node))
    print('The {} may not be available on all camera models...'.format(node))
    print('Please try a Blackfly S camera.')
    

# The following functions below are helpers for configuring image sequence captures 
def configure_sequencer_part_one(nodemap) -> bool:
    """"
    This function prepares the sequencer to accept custom configurations by
    ensuring sequencer mode is off (this is a requirement to the enabling of
    sequencer configuration mode), disabling automatic gain and exposure, and
    turning sequencer configuration mode on.
    1. Disable sequencer mode. 
    2. Turn off auto gain/ auto exposure
    3. Turn on sequencer configuration mode

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    print('*** CONFIGURING SEQUENCER ***\n')
    result = True
    try:
        
        # Ensure sequencer is off for configuration
        #
        #  *** NOTES ***
        #  In order to configure a new sequence, sequencer configuration mode
        #  needs to be turned on. To do this, sequencer mode must be disabled.
        #  However, simply disabling sequencer mode might throw an exception if
        #  the current sequence is an invalid configuration.
        #
        #  Thus, in order to ensure that sequencer mode is disabled, we first
        #  check whether the current sequence is valid. If it
        #  isn't, then we know that sequencer mode is off and we can move on;
        #  if it is, then we can manually disable sequencer mode.
        #
        #  Also note that sequencer configuration mode needs to be off in order
        #  to manually disable sequencer mode. It should be off by default, so
        #  the example skips checking this.
        #
        #  Validate sequencer configuration
        
        node_sequencer_configuration_valid = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationValid'))
        if not PySpin.IsReadable(node_sequencer_configuration_valid):
            print_retrieve_node_failure('node', 'SequencerConfigurationValid')
            return False

        sequencer_configuration_valid_yes = node_sequencer_configuration_valid.GetEntryByName('Yes')
        if not PySpin.IsReadable(sequencer_configuration_valid_yes):
            print_retrieve_node_failure('entry', 'SequencerConfigurationValid Yes')
            return False

        # If valid, disable sequencer mode; otherwise, do nothing
        node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
        if node_sequencer_configuration_valid.GetCurrentEntry().GetValue() == \
                sequencer_configuration_valid_yes.GetValue():
            if not PySpin.IsReadable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
                print_retrieve_node_failure('node', 'SequencerMode')
                return False

            sequencer_mode_off = node_sequencer_mode.GetEntryByName('Off')
            if not PySpin.IsReadable(sequencer_mode_off):
                print_retrieve_node_failure('entry', 'SequencerMode Off')
                return False

            node_sequencer_mode.SetIntValue(sequencer_mode_off.GetValue())

            print('Sequencer mode disabled...')

        # Turn off automatic exposure
        #
        #  *** NOTES ***
        #  Automatic exposure prevents the manual configuration of exposure
        #  times and needs to be turned off for this example.
        #
        #  *** LATER ***
        #  Automatic exposure is turned back on at the end of the example in
        #  order to restore the camera to its default state.
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if not PySpin.IsReadable(node_exposure_auto) or not PySpin.IsWritable(node_exposure_auto):
            print_retrieve_node_failure('node', 'ExposureAuto')
            return False

        exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
        if not PySpin.IsReadable(exposure_auto_off):
            print_retrieve_node_failure('entry', 'ExposureAuto Off')
            return False

        node_exposure_auto.SetIntValue(exposure_auto_off.GetValue())

        print('Automatic exposure disabled...')

        # Turn off automatic gain
        #
        #  *** NOTES ***
        #  Automatic gain prevents the manual configuration of gain and needs
        #  to be turned off for this example.
        #
        #  *** LATER ***
        #  Automatic gain is turned back on at the end of the example in
        #  order to restore the camera to its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if not PySpin.IsReadable(node_gain_auto) or not PySpin.IsWritable(node_gain_auto):
            print_retrieve_node_failure('node', 'GainAuto')
            return False

        gain_auto_off = node_gain_auto.GetEntryByName('Off')
        if not PySpin.IsReadable(gain_auto_off):
            print_retrieve_node_failure('entry', 'GainAuto Off')
            return False

        node_gain_auto.SetIntValue(gain_auto_off.GetValue())

        print('Automatic gain disabled...')

        # Turn configuration mode on
        #
        # *** NOTES ***
        # Once sequencer mode is off, enabling sequencer configuration mode
        # allows for the setting of each state.
        #
        # *** LATER ***
        # Before sequencer mode is turned back on, sequencer configuration
        # mode must be turned back off.
        node_sequencer_configuration_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationMode'))
        if not PySpin.IsReadable(node_sequencer_configuration_mode) or not PySpin.IsWritable(node_sequencer_configuration_mode):
            print_retrieve_node_failure('node', 'SequencerConfigurationMode')
            return False

        sequencer_configuration_mode_on = node_sequencer_configuration_mode.GetEntryByName('On')
        if not PySpin.IsReadable(sequencer_configuration_mode_on):
            print_retrieve_node_failure('entry', 'SequencerConfigurationMode On')
            return False

        node_sequencer_configuration_mode.SetIntValue(sequencer_configuration_mode_on.GetValue())

        print('Sequencer configuration mode enabled...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result

def configure_sequencer_part_two(nodemap)-> bool:
    """
    Now that the states have all been set, this function readies the camera
    to use the sequencer during image acquisition. 
    1. Turn off configuration mode

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """
    result = True

    try:
        # Turn configuration mode off
        #
        # *** NOTES ***
        # Once all desired states have been set, turn sequencer
        # configuration mode off in order to turn sequencer mode on.
        node_sequencer_configuration_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationMode'))
        if not PySpin.IsReadable(node_sequencer_configuration_mode) or not PySpin.IsWritable(node_sequencer_configuration_mode):
            print_retrieve_node_failure('node', 'SequencerConfigurationMode')
            return False

        sequencer_configuration_mode_off = node_sequencer_configuration_mode.GetEntryByName('Off')
        if not PySpin.IsReadable(sequencer_configuration_mode_off):
            print_retrieve_node_failure('entry', 'SequencerConfigurationMode Off')
            return False

        node_sequencer_configuration_mode.SetIntValue(sequencer_configuration_mode_off.GetValue())

        print('Sequencer configuration mode disabled...')

        # Turn sequencer mode on
        #
        # *** NOTES ***
        # After sequencer mode has been turned on, the camera will begin using the
        # saved states in the order that they were set.
        #
        # *** LATER ***
        # Once all images have been captured, disable the sequencer in order
        # to restore the camera to its initial state.
        node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
        if not PySpin.IsReadable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
            print_retrieve_node_failure('node', 'SequencerMode')
            return False

        sequencer_mode_on = node_sequencer_mode.GetEntryByName('On')
        if not PySpin.IsReadable(sequencer_mode_on):
            print_retrieve_node_failure('entry', 'SequencerMode On')
            return False

        node_sequencer_mode.SetIntValue(sequencer_mode_on.GetValue())

        print('Sequencer mode enabled...')

        # Validate sequencer settings
        #
        # *** NOTES ***
        # Once all states have been set, it is a good idea to
        # validate them. Although this node cannot ensure that the states
        # have been set up correctly, it does ensure that the states have
        # been set up in such a way that the camera can function.
        node_sequencer_configuration_valid = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerConfigurationValid'))
        if not PySpin.IsReadable(node_sequencer_configuration_valid):
            print_retrieve_node_failure('node', 'SequencerConfigurationValid')
            return False

        sequencer_configuration_valid_yes = node_sequencer_configuration_valid.GetEntryByName('Yes')
        if not PySpin.IsReadable(sequencer_configuration_valid_yes):
            print_retrieve_node_failure('entry', 'SequencerConfigurationValid Yes')
            return False

        if node_sequencer_configuration_valid.GetCurrentEntry().GetValue() != \
                sequencer_configuration_valid_yes.GetValue():
            print('Sequencer configuration not valid. Aborting...\n')
            return False

        print('Sequencer configuration valid...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {}'.format(ex))
        result = False

    return result

def reset_sequencer(nodemap):
    """"
    This function restores the camera to its default state by turning sequencer mode
    off and re-enabling automatic exposure and gain.

    :param nodemap: Device nodemap.
    :type nodemap: INodeMap
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Turn sequencer mode back off
        #
        # *** NOTES ***
        # The sequencer is turned off in order to return the camera to its default state.
        node_sequencer_mode = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerMode'))
        if not PySpin.IsReadable(node_sequencer_mode) or not PySpin.IsWritable(node_sequencer_mode):
            print_retrieve_node_failure('node', 'SequencerMode')
            return False

        sequencer_mode_off = node_sequencer_mode.GetEntryByName('Off')
        if not PySpin.IsReadable(sequencer_mode_off):
            print_retrieve_node_failure('entry', 'SequencerMode Off')
            return False

        node_sequencer_mode.SetIntValue(sequencer_mode_off.GetValue())

        print('Turning off sequencer mode...')

        # Turn automatic exposure back on
        #
        # *** NOTES ***
        # Automatic exposure is turned on in order to return the camera to its default state.
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsReadable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):
            exposure_auto_continuous = node_exposure_auto.GetEntryByName('Continuous')
            if PySpin.IsReadable(exposure_auto_continuous):
                node_exposure_auto.SetIntValue(exposure_auto_continuous.GetValue())
                print('Turning automatic exposure back on...')

        # Turn automatic gain back on
        #
        # *** NOTES ***
        # Automatic gain is turned on in order to return the camera to its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsReadable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
            gain_auto_continuous = node_gain_auto.GetEntryByName('Continuous')
            if PySpin.IsReadable(gain_auto_continuous):
                node_gain_auto.SetIntValue(gain_auto_continuous.GetValue())
                print('Turning automatic gain mode back on...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {} reset_sequencer '.format(ex))
        result = False

    return result

def set_single_state_from_list_of_tuple(nodemap, settings_list_of_tuple: list):

    sequence_size = len(settings_list_of_tuple)
    for sequence_number, item in enumerate(settings_list_of_tuple):
        width, height, exposure_time, gain = item
        set_single_state(nodemap, sequence_number, width, height, exposure_time, gain, sequence_size-1)

def set_single_state(nodemap, sequence_number, width_to_set, height_to_set, exposure_time_to_set, gain_to_set, max_sequence_index):
    """
    This function sets a single state. It sets the sequence number, applies
    custom settings, selects the trigger type and next state number, and saves
    the state. The custom values that are applied are all calculated in the
    function that calls this one, run_single_camera().

    :param nodemap: Device nodemap.
    :param sequence_number: Sequence number. Starts from 0. 
    :param width_to_set: Width to set for sequencer.
    :param height_to_set: Height to set fpr sequencer.
    :param exposure_time_to_set: Exposure time to set for sequencer.
    :param gain_to_set: Gain to set for sequencer.
    :type nodemap: INodeMap
    :type sequence_number: int
    :type width_to_set: int
    :type height_to_set: int
    :type exposure_time_to_set: float
    :type gain_to_set: float
    :type max_sequence: int
    :return: True if successful, False otherwise.
    :rtype: bool
    """

    try:
        result = True

        # Select the current sequence number
        #
        # *** NOTES ***
        # Select the index of the state to be set.
        #
        # *** LATER ***
        # The next state - i.e. the state to be linked to -
        # also needs to be set before saving the current state.
        node_sequencer_set_selector = PySpin.CIntegerPtr(nodemap.GetNode('SequencerSetSelector'))
        if not PySpin.IsWritable(node_sequencer_set_selector):
            print_retrieve_node_failure('node', 'SequencerSetSelector')
            return False

        node_sequencer_set_selector.SetValue(sequence_number)

        print('Setting state {}...'.format(sequence_number))

        # Set desired settings for the current state
        #
        # *** NOTES ***
        # Width, height, exposure time, and gain are set in this example. If
        # the sequencer isn't working properly, it may be important to ensure
        # that each feature is enabled on the sequencer. Features are enabled
        # by default, so this is not explored in this example.
        #
        # Changing the height and width for the sequencer is not available
        # for all camera models.
        #
        # Set width; width recorded in pixels
        node_width = PySpin.CIntegerPtr(nodemap.GetNode('Width'))
        if PySpin.IsReadable(node_width) and PySpin.IsWritable(node_width):
            width_inc = node_width.GetInc()

            if width_to_set % width_inc != 0:
                width_to_set = int(width_to_set / width_inc) * width_inc

            node_width.SetValue(width_to_set)

            print('\tWidth set to {}...'.format(node_width.GetValue()))

        else:
            print('\tUnable to set width; width for sequencer not available on all camera models...')

        # Set height; height recorded in pixels
        node_height = PySpin.CIntegerPtr(nodemap.GetNode('Height'))
        if PySpin.IsReadable(node_height) and PySpin.IsWritable(node_height):
            height_inc = node_height.GetInc()

            if height_to_set % height_inc != 0:
                height_to_set = int(height_to_set / height_inc) * height_inc

            node_height.SetValue(height_to_set)

            print('\tHeight set to %d...' % node_height.GetValue())

        else:
            print('\tUnable to set height; height for sequencer not available on all camera models...')

        # Set exposure time; exposure time recorded in microseconds
        node_exposure_time = PySpin.CFloatPtr(nodemap.GetNode('ExposureTime'))
        if not PySpin.IsReadable(node_exposure_time) or not PySpin.IsWritable(node_exposure_time):
            print_retrieve_node_failure('node', 'ExposureTime')
            return False

        exposure_time_max = node_exposure_time.GetMax()

        if exposure_time_to_set > exposure_time_max:
            exposure_time_to_set = exposure_time_max

        node_exposure_time.SetValue(exposure_time_to_set)

        print('\tExposure set to {0:.0f}...'.format(node_exposure_time.GetValue()))

        # Set gain; gain recorded in decibels
        node_gain = PySpin.CFloatPtr(nodemap.GetNode('Gain'))
        if not PySpin.IsReadable(node_gain) or not PySpin.IsWritable(node_gain):
            print_retrieve_node_failure('node', 'Gain')
            return False

        gain_max = node_gain.GetMax()

        if gain_to_set > gain_max:
            gain_to_set = gain_max

        node_gain.SetValue(gain_to_set)

        print('\tGain set to {0:.5f}...'.format(node_gain.GetValue()))

        # Set the trigger type for the current state
        #
        # *** NOTES ***
        # It is a requirement of every state to have its trigger source set.
        # The trigger source refers to the moment when the sequencer changes
        # from one state to the next.
        node_sequencer_trigger_source = PySpin.CEnumerationPtr(nodemap.GetNode('SequencerTriggerSource'))
        if not PySpin.IsReadable(node_sequencer_trigger_source) or not PySpin.IsWritable(node_sequencer_trigger_source):
            print_retrieve_node_failure('node', 'SequencerTriggerSource')
            return False

        sequencer_trigger_source_frame_start = node_sequencer_trigger_source.GetEntryByName('FrameStart')
        if not PySpin.IsReadable(sequencer_trigger_source_frame_start):
            print_retrieve_node_failure('entry', 'SequencerTriggerSource FrameStart')
            return False

        node_sequencer_trigger_source.SetIntValue(sequencer_trigger_source_frame_start.GetValue())

        print('\tTrigger source set to start of frame...')

        # Set the next state in the sequence
        #
        # *** NOTES ***
        # When setting the next state in the sequence, ensure it does not
        # exceed the maximum and that the states loop appropriately.
        final_sequence_index = max_sequence_index

        node_sequencer_set_next = PySpin.CIntegerPtr(nodemap.GetNode('SequencerSetNext'))
        if not PySpin.IsWritable(node_sequencer_set_next):
            print('Unable to select next state. Aborting...\n')
            return False

        if sequence_number == final_sequence_index:
            node_sequencer_set_next.SetValue(0)
        else:
            node_sequencer_set_next.SetValue(sequence_number + 1)

        print('\tNext state set to {}...'.format(node_sequencer_set_next.GetValue()))

        # Save current state
        #
        # *** NOTES ***
        # Once all appropriate settings have been configured, make sure to
        # save the state to the sequence. Notice that these settings will be
        # lost when the camera is power-cycled.
        node_sequencer_set_save = PySpin.CCommandPtr(nodemap.GetNode('SequencerSetSave'))
        if not PySpin.IsWritable(node_sequencer_set_save):
            print('Unable to save state. Aborting...\n')
            return False

        node_sequencer_set_save.Execute()

        print('Current state saved...\n')

    except PySpin.SpinnakerException as ex:
        print('Error: {} set_single_state'.format(ex))
        result = False

    return result

def set_cam_exposure_auto(nodemap, onOrOff: bool):

    try:
        result = True

        # Turn automatic exposure back on
        #
        # *** NOTES ***
        # Automatic exposure is turned on in order to return the camera to its default state.
        node_exposure_auto = PySpin.CEnumerationPtr(nodemap.GetNode('ExposureAuto'))
        if PySpin.IsReadable(node_exposure_auto) and PySpin.IsWritable(node_exposure_auto):

            if onOrOff:
                exposure_auto_continuous = node_exposure_auto.GetEntryByName('Continuous')
                if PySpin.IsReadable(exposure_auto_continuous):
                    node_exposure_auto.SetIntValue(exposure_auto_continuous.GetValue())
                    print('Turning automatic exposure back on...')
            else:
                exposure_auto_off = node_exposure_auto.GetEntryByName('Off')
                if PySpin.IsReadable(exposure_auto_off):
                    node_exposure_auto.SetIntValue(exposure_auto_off.GetValue())
                    print('Turning automatic exposure back off...')

    except PySpin.SpinnakerException as ex:
        print('Error: {} set_cam_exposure_auto '.format(ex))
        result = False

    return result

def set_cam_gain_auto(nodemap, onOrOff: bool):
    
    try:
        result = True

        # Turn automatic gain back on
        #
        # *** NOTES ***
        # Automatic gain is turned on in order to return the camera to its default state.
        node_gain_auto = PySpin.CEnumerationPtr(nodemap.GetNode('GainAuto'))
        if PySpin.IsReadable(node_gain_auto) and PySpin.IsWritable(node_gain_auto):
            if onOrOff:
                gain_auto_continuous = node_gain_auto.GetEntryByName('Continuous')
                if PySpin.IsReadable(gain_auto_continuous):
                    node_gain_auto.SetIntValue(gain_auto_continuous.GetValue())
                    print('Turning automatic gain mode back on...\n')
                else:
                    return False
            else:
                gain_auto_off = node_gain_auto.GetEntryByName('Off')
                if PySpin.IsReadable(gain_auto_off):
                    node_gain_auto.SetIntValue(gain_auto_off.GetValue())
                    print('Turning automatic gain mode back Off...\n')
                else:
                    return False

    except PySpin.SpinnakerException as ex:
        print('Error: {} set_cam_gain_auto '.format(ex))
        result = False

    return result

def set_cam_fps_auto(nodemap, onOrOff: bool):
    try:
        result = True

        # Turn automatic fps back on or off 
        #
        # *** NOTES ***
        # Automatic gain is turned on in order to return the camera to its default state.
        node_fps_auto = PySpin.CEnumerationPtr(nodemap.GetNode('AcquisitionFrameRateEnable'))
        if PySpin.IsReadable(node_fps_auto) and PySpin.IsWritable(node_fps_auto):
            if onOrOff:
                fps_auto_on = node_fps_auto.GetEntryByName('On')
                if PySpin.IsReadable(fps_auto_on):
                    node_fps_auto.SetIntValue(fps_auto_on.GetValue())
                    print('Turning automatic FPS mode back on...\n')
                else:
                    return False
            else:
                fps_auto_off = node_fps_auto.GetEntryByName('Off')
                if PySpin.IsReadable(fps_auto_off):
                    node_fps_auto.SetIntValue(fps_auto_off.GetValue())
                    print('Turning automatic FPS mode back Off...\n')
                else:
                    return False

    except PySpin.SpinnakerException as ex:
        print('Error: {} set_cam_fps_auto '.format(ex))
        result = False

    return result
