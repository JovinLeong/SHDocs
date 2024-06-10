import os
import sys
import cv2
import base64
import numpy as np

sys.path.append("../..")
from utils.common import get_timenow
from utils.constants import IMAGE_INDEX_MAPPING

def b64str_to_image(b64_str):
    decoded_data = base64.b64decode(b64_str)
    np_data = np.frombuffer(decoded_data, np.uint8)
    return cv2.imdecode(np_data, cv2.IMREAD_UNCHANGED)

def path_to_b64str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string.decode('utf-8')

def numpy_to_b64(np_image):
    return base64.b64encode(np_image)

def b64_to_numpy(b64):
    r = base64.decodebytes(b64)
    return np.frombuffer(r, dtype=np.float64)

def draw_predictions(predictions, draw_img):

    # Convert image to BGR if not gray
    if len(draw_img.shape) == 2:
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

    if draw_img.shape[2] == 1:
        draw_img = cv2.cvtColor(draw_img, cv2.COLOR_GRAY2BGR)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    txt_color = (255, 0, 0)
    txt_thickness = 1
    box_thickness = 2
    box_color = (0, 255, 0)

    # Iterate over predictions
    for box_info in predictions:
        
        # Draw boxes
        box_coordinates = box_info['box']
        box_score = box_info['score']
        draw_img = cv2.line(draw_img, (int(box_coordinates[0][0]),  int(box_coordinates[0][1]) ) , (int(box_coordinates[1][0]),  int(box_coordinates[1][1] )) , box_color, box_thickness)
        draw_img = cv2.line(draw_img, (int(box_coordinates[1][0]),  int(box_coordinates[1][1]) ) , (int(box_coordinates[2][0]),  int(box_coordinates[2][1] )) , box_color, box_thickness)
        draw_img = cv2.line(draw_img, (int(box_coordinates[2][0]),  int(box_coordinates[2][1]) ) , (int(box_coordinates[3][0]),  int(box_coordinates[3][1] )) , box_color, box_thickness)
        draw_img = cv2.line(draw_img, (int(box_coordinates[3][0]),  int(box_coordinates[3][1]) ) , (int(box_coordinates[0][0]),  int(box_coordinates[0][1] )) , box_color, box_thickness)

        # Draw score
        box_score_str = f'{box_score:.3f}'
        org = (int(box_coordinates[0][0]),  int(box_coordinates[0][1] - 10))
        draw_img = cv2.putText(draw_img, box_score_str, org, font, 
                    fontScale, txt_color, txt_thickness, cv2.LINE_AA)
    return draw_img

def count_detected_text_area(predictions, total_area_of_detected_text=0.0):

    for box_info in predictions['detected_texts']:
        box_coordinates = box_info['box']
        vec1 = box_coordinates[1] - box_coordinates[0]
        vec2 = box_coordinates[3] - box_coordinates[0]
        total_area_of_detected_text += np.linalg.norm( np.cross( [vec1[0], vec1[1], 0], [vec2[0], vec2[1], 0]) )
    return total_area_of_detected_text

def save_all_captured_images(input_images, output_dir='../data/captures'):
    output_folder = os.path.join(output_dir, "raw_" + get_timenow())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Iterate across input image arrays and write images as per their setting
    for index, image in enumerate(input_images):
        cv2.imwrite(f'{output_folder}/{IMAGE_INDEX_MAPPING[index]}.png', image)
    return None

def save_images_to_folder(input_image):
    output_folder = os.path.join('output', get_timenow())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for index, image in enumerate(input_image):
        cv2.imwrite(f"{output_folder}/image_{str(index)}.png", image) 
    return None

def save_image_to_folder(input_image, suffix=''):
    output_folder = os.path.join('output', get_timenow())
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_name = f"{output_folder}/image_{get_timenow()}_{suffix}.png"
    cv2.imwrite(output_name, input_image)
    return output_name
