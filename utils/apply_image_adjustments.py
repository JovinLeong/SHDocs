import os
import cv2
import json
import argparse
import numpy as np

def document_scale(document, scale_settings):    
    return cv2.resize(document, (int(document.shape[1] * scale_settings["scale_factor"]), 
                                 int(document.shape[0] * scale_settings["scale_factor"])))

def document_horizontal_crop(document, scale_settings):
    return document[:, scale_settings["horizontal_crop_factor"]:]

def document_vertical_crop(document, scale_settings):
    return document[scale_settings["vertical_crop_factor"]:, :]

def document_end_horizontal_crop(document, scale_settings):
    return document[:, :scale_settings["end_horizontal_crop_factor"]]

def document_end_vertical_crop(document, scale_settings):
    return document[: scale_settings["end_vertical_crop_factor"], :]

def document_pad(document, scale_settings):
    return np.pad(document, scale_settings["pad_factor"])

def document_rotate(document, scale_settings):
    # Rotate document accordingly
    if not scale_settings["no_crop_rotate"]:
        center = (document.shape[1] / 2, document.shape[0] / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, scale_settings["rotate_factor"], 1)
        return cv2.warpAffine(document, rotation_matrix, (document.shape[1], document.shape[0]))
    elif scale_settings["rotate_factor"] > 60:
        document = np.rot90(document)            
        center = (document.shape[1] / 2, document.shape[0] / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, scale_settings["rotate_factor"] - 90, 1)
        return cv2.warpAffine(document, rotation_matrix, (document.shape[1], document.shape[0]))
    elif scale_settings["rotate_factor"] < -60:
        document = np.rot90(document, k=3)
        center = (document.shape[1] / 2, document.shape[0] / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, scale_settings["rotate_factor"] + 90, 1)
        return cv2.warpAffine(document, rotation_matrix, (document.shape[1], document.shape[0]))
    else:
        raise Exception("No crop rotation not supported for this angle")

def transform_image_from_scale_setting(image, scale_settings):
    
    # Apply transformations
    scaled_document = image.copy()
    print("\nScale settings: ", scale_settings)
    scaled_document = document_rotate(scaled_document, scale_settings)
    scaled_document = document_scale(scaled_document, scale_settings)
    scaled_document = document_pad(scaled_document, scale_settings)
    scaled_document = document_horizontal_crop(scaled_document, scale_settings)
    scaled_document = document_vertical_crop(scaled_document, scale_settings)
    scaled_document = document_end_horizontal_crop(scaled_document, scale_settings)
    scaled_document = document_end_vertical_crop(scaled_document, scale_settings)
    return scaled_document

def transform_images_from_metadata(metadata, image_directory, output_directory):
    
    # Load metadata
    with open(metadata, 'r') as f:
        metadata = json.load(f)
    
    # Get images
    image_names = os.listdir(image_directory)
    
    for image_name in image_names:
        
        if image_name[0] == '.':
            continue
        
        # Load image
        image = cv2.imread(os.path.join(image_directory, f'{image_name}'), cv2.IMREAD_GRAYSCALE)
        
        # Get scale settings for image
        scale_settings = metadata['documents'][image_name[:-4]]['scale_settings']
        
        # Apply transformations
        transformed_image = transform_image_from_scale_setting(image, scale_settings)
        
        # Output image
        output_path = os.path.join(output_directory, f'{image_name}')
        cv2.imwrite(output_path, transformed_image)
        print(f"{image_name} transformed and saved to {output_path}")
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Transform images based on metadata')
    parser.add_argument('--metadata', '-m', type=str, help='Path to metadata file')
    parser.add_argument('--image_directory', '-i', type=str, help='Path to source image directory')
    parser.add_argument('--output_directory', '-o', type=str, help='Path to output image directory')
    args = parser.parse_args()
    
    transform_images_from_metadata(args.metadata, args.image_directory, args.output_directory)
