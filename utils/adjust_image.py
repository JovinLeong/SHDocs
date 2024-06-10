import os
import cv2
import json
import argparse
import numpy as np
from datetime import datetime

class DocumentScaler:
    
    def __init__(self, raw_document_directory, annotation_filepath, processed_document_directory, scale_settings, metadata_json_filepath):
        
        self.no_crop_rotate = False

        # Update with scale settings
        for key, value in scale_settings.items():
            setattr(self, key, value)
        
        self.raw_document_directory = raw_document_directory
        self.annotation_filepath = annotation_filepath
        self.processed_document_directory = processed_document_directory
        self.metadata_json_filepath = metadata_json_filepath
        
        # Load annotation
        self.annotation = self.load_annotation()
        
        # Obtain all images in the folder and representative document image
        self.document_images = os.listdir(self.raw_document_directory)
        self.representative_document_filepath = f'{raw_document_directory}/{self.document_images[0]}'
        self.representative_document = self.load_document(self.representative_document_filepath)
        
    def load_document(self, document_filepath):
        # Load document image
        return cv2.imread(document_filepath, cv2.IMREAD_GRAYSCALE)

    def load_annotation(self):
        # Load corresponding annotation
        with open(self.annotation_filepath, 'r') as f:
            annotation = json.load(f)
        return annotation

    def add_annotation_overlay(self):
        
        # Instantiate resultant document
        overlay = self.representative_document.copy()
        overlay = self.document_rotate(overlay)
        overlay = self.document_scale(overlay)
        overlay = self.document_pad(overlay)
        overlay = self.document_horizontal_crop(overlay)
        overlay = self.document_vertical_crop(overlay)
        overlay = self.document_end_horizontal_crop(overlay)
        overlay = self.document_end_vertical_crop(overlay)

        # Iterate across annotations
        for bbox in self.annotation['form']:
            
            # Get bbox coordinates
            x0, y0, x1, y1 = bbox['box']
            
            cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), 2)
            
        return overlay

    def document_scale(self, document):
        
        # Scale document accordingly
        return cv2.resize(document, 
                          (int(document.shape[1] * self.scale_factor), 
                           int(document.shape[0] * self.scale_factor)))

    def document_horizontal_crop(self, document):
        
        return document[:, self.horizontal_crop_factor:]
    
    def document_vertical_crop(self, document):
        
        return document[self.vertical_crop_factor:, :]

    def document_end_horizontal_crop(self, document):
        
        return document[:, :self.end_horizontal_crop_factor]
    
    def document_end_vertical_crop(self, document):
        
        return document[:self.end_vertical_crop_factor, :]
    
    def document_pad(self, document):
        
        # Pad document; padding_factor: ((0, 0), (0, 0))
        return np.pad(document, self.pad_factor)

    def document_rotate(self, document):
        
        # Rotate document accordingly
        if not self.no_crop_rotate:
            center = (document.shape[1] / 2, document.shape[0] / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate_factor, 1)
            return cv2.warpAffine(document, rotation_matrix, (document.shape[1], document.shape[0]))
        elif self.rotate_factor > 60:
            document = np.rot90(document)            
            center = (document.shape[1] / 2, document.shape[0] / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate_factor - 90, 1)
            return cv2.warpAffine(document, rotation_matrix, (document.shape[1], document.shape[0]))
        elif self.rotate_factor < -60:
            document = np.rot90(document, k=3)
            center = (document.shape[1] / 2, document.shape[0] / 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate_factor + 90, 1)
            return cv2.warpAffine(document, rotation_matrix, (document.shape[1], document.shape[0]))
        else:
            raise Exception("No crop rotation not supported for this angle")

    def set_rotate_factor(self, rotate_factor):
        self.rotate_factor = rotate_factor
        return None
    
    def set_no_crop_rotate(self):
        self.no_crop_rotate = not self.no_crop_rotate
        return None

    def set_scale_factor(self, scale_factor):
        self.scale_factor = scale_factor
        return None

    def set_pad_factor(self, pad_factor):
        self.pad_factor = pad_factor
        return None

    def set_horizontal_crop_factor(self, crop_factor):
        self.horizontal_crop_factor = crop_factor
        return None

    def set_vertical_crop_factor(self, crop_factor):
        self.vertical_crop_factor = crop_factor
        return None
    
    def get_scale_settings(self):
        
        scale_settings = {
            'rotate_factor': self.rotate_factor,
            'scale_factor': self.scale_factor,
            'pad_factor': self.pad_factor,
            'horizontal_crop_factor': self.horizontal_crop_factor,
            'vertical_crop_factor': self.vertical_crop_factor
        }
        
        return scale_settings

    def _update_dataset_metadata(self, document_metadata):
        
        # If file doesnt exist, create it
        if not os.path.exists(self.metadata_json_filepath):
            metadata = {
                'documents': {}
            }
            
        else:
            with open(self.metadata_json_filepath, 'r') as f:
                metadata = json.load(f)
        
        # Access the documents in the JSON, update, and write
        metadata['documents'].update(document_metadata)
        with open(self.metadata_json_filepath, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return None

    def generate_scaled_document(self, document_filename):
        
        # Specify output filepath
        raw_document_filepath = os.path.join(self.raw_document_directory, document_filename)
        processed_document_filepath = os.path.join(self.processed_document_directory, document_filename)
        
        # Load document
        scaled_document = self.load_document(raw_document_filepath)
        
        # Scale document
        scaled_document = self.document_rotate(scaled_document)
        scaled_document = self.document_scale(scaled_document)
        scaled_document = self.document_pad(scaled_document)
        scaled_document = self.document_horizontal_crop(scaled_document)
        scaled_document = self.document_vertical_crop(scaled_document)
        scaled_document = self.document_end_horizontal_crop(scaled_document)
        scaled_document = self.document_end_vertical_crop(scaled_document)
        
        
        # Save scaled document image
        cv2.imwrite(processed_document_filepath, scaled_document)
        return None

    def generate_all_scaled_documents(self):
        
        # Make directory if it doesn't exist
        os.makedirs(self.processed_document_directory, exist_ok=True)
        
        # Get generate all scaled documents in the directory
        for document_image in self.document_images:
            self.generate_scaled_document(document_filename=document_image)
        
        # Update metadata
        document_id = self.raw_document_directory.split('/')[-1]
        document_metadata = {
            document_id: {
                'timestamp': str(datetime.now()).split('.')[0].replace('-','').replace(' ', '_').replace(':', ''),
                'scale_settings': {
                        'rotate_factor': float(self.rotate_factor),
                        'no_crop_rotate': self.no_crop_rotate,
                        'scale_factor': float(self.scale_factor),
                        'pad_factor': self.pad_factor,
                        'horizontal_crop_factor': int(self.horizontal_crop_factor),
                        'vertical_crop_factor': int(self.vertical_crop_factor),
                        'end_horizontal_crop_factor': int(self.end_horizontal_crop_factor),
                        'end_vertical_crop_factor': int(self.end_vertical_crop_factor)
                    },
                'raw_document_directory': self.raw_document_directory,
                'processed_document_filepath': self.processed_document_directory,
            }
        }
        self._update_dataset_metadata(document_metadata)
        return None
    
def get_document_id(document_name):
    
    if '_' in document_name[-2]:
        return '_'.join(document_name.split('_')[:-1])
    else:
        return document_name

def get_default_scale_settings(document, all_documents, default_scale_settings_filepath):
    
    past_documents = [document_prefix for document_prefix in all_documents.keys() if (document == document_prefix[:-2]) or (document == document_prefix)]

    # Use previous scale settings for the document if its available
    if document_name in documents:
        print("Using previous scale setting")
        scale_settings = documents[document_name]['scale_settings']
    
    # Else, use one of the past documents
    elif len(past_documents) > 0:
        print("Using similar document scale setting")
        past_documents.sort()
        scale_settings = documents[past_documents[-1]]['scale_settings']
    
    # If previous scale settings don't exist, load defaults
    else:
        print("Using default scale settings")
        with open(default_scale_settings_filepath, 'r') as f:
            default_scale_settings = json.load(f)['scale_settings']    
        scale_settings = default_scale_settings
    
    return scale_settings

if __name__ == '__main__':
    
    # Required user inputs
    parser = argparse.ArgumentParser(description='Scale document captures to align with annotations')
    parser.add_argument('-d', '--document_name', type=str, required=True, help='Path to document capture directory')
    
    # Add optional settings - note that the defaults might need to be overwritten depending on where your files reside
    parser.add_argument('--raw_document_directory', type=str, required=False, help='Where the raw input captured document images from the camera reside', default='../data/shdocs_dataset/raw_captures/training_data/images')
    parser.add_argument('--processed_document_directory', type=str, required=False, help='Where the processed output document images from the camera reside', default='../data/shdocs_dataset/processed_captures/training_data/images')
    parser.add_argument('--annotation_directory', type=str, required=False, help='', default='../data/shdocs_dataset/processed_captures/training_data/annotations')
    parser.add_argument('--metadata_path', type=str, required=False, help='', default="../data/shdocs_dataset/image_adjustment_metadata.json")
    parser.add_argument('--default_scale_settings', type=str, required=False, help='Where the default scaling resides', default='./image_adjustment_metadata/default_scale_settings.json')
    
    # CLI
    args = parser.parse_args()
    document_name = args.document_name
    document_id = get_document_id(document_name)
    default_scale_settings_filepath = args.default_scale_settings
    raw_document_directory = f"{args.raw_document_directory}/{document_name}"
    processed_document_directory = f"{args.processed_document_directory}/{document_name}"
    annotation_filepath = f"{args.annotation_directory}/{document_id}.json"
    scaling_metadata_filepath = args.metadata_path
    
    # Load metadata
    if not os.path.exists(scaling_metadata_filepath):
        metadata = {
            'documents': {}
        }
        
    else:
        with open(scaling_metadata_filepath, 'r') as f:
            metadata = json.load(f)
    
    # Access the documents in the JSON and get scale settings
    documents = metadata['documents']
    scale_settings = get_default_scale_settings(document_id, documents, default_scale_settings_filepath)

    # Instantiate scaler with initial overlay
    document_scaler = DocumentScaler(raw_document_directory=raw_document_directory, 
                                     annotation_filepath=annotation_filepath, 
                                     processed_document_directory=processed_document_directory,
                                     scale_settings=scale_settings, 
                                     metadata_json_filepath=scaling_metadata_filepath)
    representative_document_image = document_scaler.add_annotation_overlay()

    # Scaling interface
    window_name = 'Representative document scaler'
    
    while True:
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            cv2.destroyAllWindows()
            print(document_scaler.get_scale_settings())
            break
            
        elif key == ord('q'):
            cv2.destroyAllWindows()
            document_scaler.set_rotate_factor(document_scaler.rotate_factor - 0.1)
    
        elif key == ord('e'):
            cv2.destroyAllWindows()
            document_scaler.set_rotate_factor(document_scaler.rotate_factor + 0.1)
            
        elif key == ord('w'):
            cv2.destroyAllWindows()
            document_scaler.set_vertical_crop_factor(document_scaler.vertical_crop_factor + 1)
    
        elif key == ord('s'):
            cv2.destroyAllWindows()
            document_scaler.set_vertical_crop_factor(np.max((document_scaler.vertical_crop_factor - 1, 0)))
            
        elif key == ord('d'):
            cv2.destroyAllWindows()
            document_scaler.set_horizontal_crop_factor(np.max((document_scaler.horizontal_crop_factor - 1, 0)))
            
        elif key == ord('a'):
            cv2.destroyAllWindows()
            document_scaler.set_horizontal_crop_factor(document_scaler.horizontal_crop_factor + 1)
            
        elif key == ord('r'):
            cv2.destroyAllWindows()
            document_scaler.set_scale_factor(document_scaler.scale_factor + 0.001)
            
        elif key == ord('f'):
            cv2.destroyAllWindows()
            document_scaler.set_scale_factor(document_scaler.scale_factor - 0.001)
            
        elif key == ord('t'):
            cv2.destroyAllWindows()
            document_scaler.set_no_crop_rotate()
            
        elif key == 13:
            cv2.destroyAllWindows()
            document_scaler.generate_all_scaled_documents()
            break
        
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 700, 900) 
        # cv2.moveWindow(window_name, 900, -900)
        cv2.imshow(window_name, document_scaler.add_annotation_overlay())
        