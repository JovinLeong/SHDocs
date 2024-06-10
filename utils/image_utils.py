import os
import cv2
import sys
import shutil

def duplicate_annotations(annotation_dir, output_dir):
    """Duplicate annotations for each alphabet which corresponds to the different SHDocs transparency
    Parameters:
        annotation_dir (str): Path to annotations directory
        output_dir (str): Path to output directory
        
    Sample usage:
        parser = argparse.ArgumentParser(description='Process captures to properly sort and rename into dataset')
        parser.add_argument('--annotation_dir', type=str, required=True, help='Path to annotations directory')
        parser.add_argument('--output_dir', type=str, required=True, help='Path to output directory')

        args = parser.parse_args()
        duplicate_annotations(args.annotation_dir, args.output_dir)
    """    
    # Get annotations
    annotations = os.listdir(annotation_dir)
    
    # Iterate across capture folders
    for annotation in annotations:
        
        # Skip .DS_Store
        if annotation[0] == '.':
            continue
        
        # Get annotation path
        annotation_path = os.path.join(annotation_dir, annotation)
        
        # Iterate across alphabets
        for alphabet in ALPHABET_MAPPING.values():
            
            # Create annotation
            annotation_name = f"{annotation.split('.')[0]}_{alphabet}.json"
            shutil.copy2(annotation_path, os.path.join(output_dir, annotation_name))
    
    return None


def rename_document_images(capture_directory, sorted_capture_directory):
    """Rename document images to include the capture folder name
    Parameters:
        capture_directory (str): Path to the raw captured document images
        sorted_capture_directory (str): Path to the sorted captured document images
        
    Sample usage:
        parser = argparse.ArgumentParser(description='Process captures to properly sort and rename into dataset')
        parser.add_argument('--capture_path', type=str, required=True, help='Path to raw captured document images')
        parser.add_argument('--sorted_capture_path', type=str, required=True, help='Path to sorted captured document images')

        args = parser.parse_args()
        rename_document_images(args.capture_path, args.sorted_capture_path)
    """
    
    # Get folders in the capture directory and sorted captures directory
    capture_folders = os.listdir(capture_directory)
    sorted_capture_folders = os.listdir(sorted_capture_directory)
    
    # Iterate across capture folders
    for capture_folder in capture_folders:
        
        # If capture folder has already been sorted, skip
        if capture_folder in sorted_capture_folders or capture_folder[0] == '.':
            continue
        
        # Else, create the folder
        os.mkdir(os.path.join(sorted_capture_directory, capture_folder))
        
        # Get captures from the capture folder
        captures = os.listdir(os.path.join(capture_directory, capture_folder))
        for capture in captures:
            
            new_capture_name = f"{capture_folder}_{capture}"
            shutil.copy2(os.path.join(capture_directory, capture_folder, capture), os.path.join(sorted_capture_directory, capture_folder, new_capture_name))
    
    return None


def convert_and_save_to_rgb(input_path, output_path):
    """
    Convert a grayscale image to an RGB image with three channels and save it to a file.
    
    Parameters:
        input_path (str): Path to the input grayscale image file.
        output_path (str): Path to save the output RGB image file.
    
    Sample usage:
    image_path = sys.argv[1]
    for image in os.listdir(image_path):
        
        if image[0] != '.':
            convert_and_save_to_rgb(os.path.join(image_path, image), os.path.join(image_path, image))
    """
    # Read the grayscale image
    gray_image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    
    # Convert grayscale image to RGB by duplicating the single channel
    rgb_image = cv2.merge([gray_image, gray_image, gray_image])
    
    # Save the RGB image to the output path
    cv2.imwrite(output_path, rgb_image)
    
def pad_image_to_multiple(image, crop_size):
    """Pad an image to be divisible by the crop size"""
    h, w = image.shape[0], image.shape[1]
    pad_h = (crop_size[0] - h % crop_size[0]) % crop_size[0]
    pad_w = (crop_size[1] - w % crop_size[1]) % crop_size[1]
    padded_image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
    
    return padded_image

def tumbling_window(image, crop_size):
    """Take in an image and perform tumbling crops"""

    # Pad image to be divisible by the stride
    crops = []
    for i in range(0, image.shape[0] - crop_size[0] + 1, crop_size[0]):
        for j in range(0, image.shape[1] - crop_size[1] + 1, crop_size[1]):
            crop = image[i:i + crop_size[0], j:j + crop_size[1]]
            crops.append(crop)
    return crops


def tumble_crop_images(source_directory, output_directory, crop_size):
    """Tumble crop images into smaller crops.
    
    Parameters:
        source_directory (str): Path to the source directory containing the images.
        output_directory (str): Path to the output directory to save the cropped images.
        crop_size (tuple): Size of the crops to create.
        
    Sample usage:
    tumble_crop_images(source_directory='../data/shdocs_dataset/raw_captures/training_data/raw_captures_restructured', 
                       output_directory='../data/shdocs_dataset/mimo_training_data', crop_size=(256, 256))
    """
    
    # Get images
    s0_norm_images = os.listdir(os.path.join(source_directory, 's0_norm'))
    # deglared_images = os.listdir(os.path.join(source_directory, 'deglared'))
    
    # Iterate across images
    for image in s0_norm_images:
        
        # Pad image to be divisible by the stride
        s0_norm_image = cv2.imread(os.path.join(source_directory, 's0_norm', image))
        padded_image = pad_image_to_multiple(s0_norm_image, crop_size)
        image_crops = tumbling_window(padded_image, crop_size)
        
        # Save crops
        for i, crop in enumerate(image_crops):
            cv2.imwrite(os.path.join(output_directory, 's0_norm', f"{image.split('.')[0]}_{i}.png"), crop)
        
        deglared_image = cv2.imread(os.path.join(source_directory, 'deglared', image))
        padded_image = pad_image_to_multiple(deglared_image, crop_size)
        image_crops = tumbling_window(padded_image, crop_size)
        
        for i, crop in enumerate(image_crops):
            cv2.imwrite(os.path.join(output_directory, 'deglared', f"{image.split('.')[0]}_{i}.png"), crop)
        
    return None


def restructure_captures(processed_captures_path, processed_captures_restructured_path, image_type):
    """Restructure the processed captures
    
    Parameters:
        processed_captures_path (str): Path to the processed captures
        processed_captures_restructured_path (str): Path to the restructured processed captures
        image_type (str): Type of image to restructure
        
    Sample usage:
        parser = argparse.ArgumentParser(description='Restructure the processed captures')
        parser.add_argument('--processed_captures_path', type=str, required=False, default='../data/shdocs_dataset/processed_captures/testing_data/images')
        parser.add_argument('--processed_captures_restructured_path', type=str, required=False, default='../data/shdocs_dataset/restructured/testing_data')
        parser.add_argument('--image_type', type=str, required=True)
        args = parser.parse_args()
        
        restructured_captures(args.processed_captures_path, args.processed_captures_restructured_path, args.image_type)
    """
    
    # Get all the folders in the processed captures path
    processed_captures = os.listdir(processed_captures_path)
    
    if image_type == 'camera':
        
        # Within each folder, get the s0_norm image
        for processed_capture in processed_captures:
                
            if processed_capture[0] == '.' or processed_capture[-2] == '_':
                continue
            
            # Get image path
            image_path = os.path.join(processed_captures_path, processed_capture, f"s0_norm.png")
            
            # Move the image to the processed_captures_restructured_path with image name
            shutil.copy2(image_path, os.path.join(processed_captures_restructured_path, image_type, f"{processed_capture}.png"))
        
    else:    
        # Within each folder, get the s0_norm image
        for processed_capture in processed_captures:
            
            if processed_capture[0] == '.' or processed_capture[-2] != '_':
                continue
            
            # Get image path
            image_path = os.path.join(processed_captures_path, processed_capture, f"{image_type}.png")
            
            if not os.path.exists(os.path.join(processed_captures_restructured_path, image_type)):
                os.makedirs(os.path.join(processed_captures_restructured_path, image_type))
            
            # Move the image to the processed_captures_restructured_path with image name
            shutil.copy2(image_path, os.path.join(processed_captures_restructured_path, image_type, f"{processed_capture}.png"))
        
    return None


