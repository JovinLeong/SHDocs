# Polarcam

This directory contains the code and instructions to use the application and GUI designed for collecting specular highlight data. Our code was developed to operate with a FLIR Blackfly S camera equipped with a Sony IMX250MZR, CMOS, 2/3" polarization sensor (BFS-U3-51S5P). Although it can be modified to work with other systems, these will not be supported by us.

Certain components are based on the FLIR Spinnaker SDK [linked here for Asia](https://www.flir.asia/products/spinnaker-sdk), [here for Europe](https://www.flir.eu/products/spinnaker-sdk), and [here for the rest of the world](https://www.flir.com/products/spinnaker-sdk/).

## Setup

### Spinnaker SDK installation

1. Download the appropriate FLIR Spinnaker SDK for your machine. You will need to download and subsequently install the SDK, drivers, and the Python library itself.

    The packages are [linked here for Asia](https://www.flir.asia/products/spinnaker-sdk), [here for Europe](https://www.flir.eu/products/spinnaker-sdk), and [here for the rest of the world](https://www.flir.com/products/spinnaker-sdk/). Account creation might be required to download the SDK.

    In our implementation, we downloaded the following versions to run on an x86-64 Linux environment with Python 3.10:

    ```{bash}
    spinnaker-3.1.0.79-amd64-pkg.tar.gz

    spinnaker_python-3.1.0.79-cp38-cp38-linux_x86_64.tar.gz
    ```

2. After downloading the SDK, following the installation instructions provided in the ```README``` file within the SDK to install the Spinnaker library and the appropriate drivers for your machine.

    In our implementation, we used the installation instructions in the following file:

    ```{bash}
    spinnaker-3.1.0.79-amd64-pkg/spinnaker-3.1.0.79-amd64/README
    ```

3. Next, follow the instructions in the ```Readme.txt``` file in the Spinnaker Python library to install the Python library.

    In our implementation, we used the installation instructions in the following file:

    ```{bash}
    spinnaker_python-3.1.0.79-cp38-cp38-linux_x86_64/Readme.txt
    ```

### PyQt5 setup

Next, you will need to install the libraries required to run PyQt5 on which the GUI runs.

Execute the following setup scripts to install the required libraries for Ubuntu. You will need to identify and run the equivalent install commands for your package manager if you are not using Ubuntu APT.

```{bash}
sudo apt update -q
sudo apt install -y -q build-essential libgl1-mesa-dev

sudo apt install -y -q libxkbcommon-x11-0
sudo apt install -y -q libxcb-image0
sudo apt install -y -q libxcb-keysyms1
sudo apt install -y -q libxcb-render-util0
sudo apt install -y -q libxcb-xinerama0
sudo apt install -y -q libxcb-icccm4

sudo apt install -y qttools5-dev-tools
sudo apt install -y qttools5-dev
```

### Python environment setup

We recommend creating a virtual environment using ```venv``` or ```conda``` to manage your Python libraries and dependencies.

Update the ```requirements.txt``` file such that ```spinnaker-python``` points to the exact directory where the ```spinnaker-python.whl``` file is located.

Next, install the dependencies using:

```{bash}
pip install requirements.txt
```

## Using the GUI

1. Activate your virtual environment with the installed dependencies.

2. Ensure that your camera is connected to your machine.

3. Run the GUI in the terminal:

    ```{python}
    python gui/controller/qt_polarcam_controller.py
    ```

### Camera preview

Within the GUI, click on the "Start Camera Preview" button to start the streamed camera preview where you should be able to see the deglared image and the $S_0$ reconstruction.

### Image capture

Within the GUI, click on the "Capture" button to start capture the image and store it to the default directory. For each capture, the deglared, $S_0$, $i_{0}$, $i_{45}$, $i_{90}$, $i_{135}$ should be captured all at once.

## Modifying the scripts

To modify the main wrapper class, you can make changes to ```PolarCam.py```

To initialize the camera:

```{python}
import FLIRPolarCam.PolarCam

polar_cam = FLIRPolarCam.PolarCam()
polar_cam.start_acquisition()
```

To capture a frame in polarized8 format:

```{python}
image_result = polar_cam.grab_image()
```

To convert a frame to the respective filtered images and deglared output in NumPy:

```{python}
image_polarized_i0, image_polarized_i45, image_polarized_i90, image_polarized_i135, image_dolp, image_deglared = polar_cam.grab_all_polarized_image(image_result)
```

To append all the images to a single image:

```python
image_display = FLIRPolarCam.PolarCam.append_images_to_panel(image_polarized_i0, image_polarized_i45, image_polarized_i90, image_polarized_i135, image_dolp, image_deglared)
```

## Modifying the GUI

The Qt GUI files are located in the ```gui/qt_ui``` directory.

If any modifications to the UI files were made using tools such as Qt Designer, you will need to generate the Python equivalent source file.

Execute the following command in the ```gui/qt_ui``` directory:

```{bash}
pyuic5 polarcam.ui -o ../generated/ui_polarcam.py
```
