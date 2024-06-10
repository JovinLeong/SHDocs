import io
import os
import argparse
import numpy as np
from PIL import Image
from datetime import datetime
import matplotlib.pyplot as plt
from utils.constants import FUNSD_EVAL_SET, ALPHABET_SET
from utils.common import dump_dict_to_json

# Inference functions
import boto3
import easyocr
import pytesseract
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes, OperationStatusCodes
from azure.ai.formrecognizer import FormRecognizerClient, DocumentAnalysisClient

def concatenate_images_vertically(image_files, image_dir, margin=10):
    
    # Crop range dict
    crop_metadata = {}
    
    # Load the images
    images = [Image.open(os.path.join(image_dir, i)) for i in image_files]
    
    # Get the dimensions of the images
    widths, heights = zip(*(i.size for i in images))
    
    # Get the total height
    total_height = sum(heights) + len(images) * margin
    
    # Create a new image with the width of the widest image and the total height
    concatenated_image = Image.new('RGB', (max(widths), total_height))
    
    # Paste the images
    y_offset = 0
    for i, im in enumerate(images):
        
        w, h  = im.size
        concatenated_image.paste(im, (0, y_offset))
        
        # Update metadata
        crop_metadata[i] = {
            'image': image_files[i],
            'range': (y_offset, y_offset + h),
            'text': ''
        }
        
        # Update offset
        y_offset += im.size[1] + margin
        
    return concatenated_image, crop_metadata

def check_in_range(source_range, target_range, margin=10):
    halved_margin = margin // 2
    return source_range[0] >= target_range[0] - halved_margin and source_range[1] <= target_range[1] + halved_margin

def pad_to_50(image):
    
    # If image is smaller than 50x50, pad it with zeros
    h, w = image.shape
    if h < 50:
        image = np.pad(image, ((0, 50-h), (0, 0)), mode='constant', constant_values=0)
    if w < 50:
        image = np.pad(image, ((0, 0), (0, 50-w)), mode='constant', constant_values=0)
    return (image * 255).astype(np.uint8)

def align_document_intelligence_results(results, crop_metadata, margin):
    
    # Metadata index
    metadata_index = 0
    
    # Iterate over the results
    for line in results[0].lines:
    
        # Get min and max y coordinates
        min_y = np.min(np.array(line.bounding_box)[:, 1]).astype(int)
        max_y = np.max(np.array(line.bounding_box)[:, 1]).astype(int)

        # Iterate across metadata
        for key, value in list(crop_metadata.items())[metadata_index:]:

            # If the y coordinate is within the range, update the metadata text
            if check_in_range((min_y, max_y), value['range'], margin=margin):
                value['text'] += line.text
                
                # Set metadata index to current index and break out of loop
                metadata_index = key
                break
    
    return crop_metadata

def document_intelligence_batched_inference(concatenated_image, concatenation_metadata, margin):
    
    # Call model inference
    image_stream = io.BytesIO()
    concatenated_image.save(image_stream, format='PNG')
    image_stream.seek(0)

    try: 
        poller = form_recognizer_client.begin_recognize_content(image_stream)
        complete_document_result = poller.result()

        aligned_predictions = align_document_intelligence_results(complete_document_result, concatenation_metadata, margin)
        
        # Restructure results and update outputs
        predictions = {}
        for key, value in aligned_predictions.items():
            
            predictions[value['image']] = [
                {
                    'Text': value['text']
                }
            ]
    
    except Exception as e:
        print("Error: ", e)
        predictions = {}
        for key, value in concatenation_metadata.items():
            predictions[value['image']] = [
                {
                    'Text': value['text']
                }
            ]        
    return predictions

def align_textract_results(results, crop_metadata, concatenated_image_dims, margin):
    
    # Extract detected text from the response
    detected_text = [block for block in results['Blocks'] if block['BlockType'] in ['LINE']]    
    metadata_index = 0
    detected_lines = {}
    
    # Iterate over the results and concatenate text
    for line in detected_text:
        
        y0 = int(line['Geometry']['BoundingBox']['Top'] * concatenated_image_dims[1])
        y1 = y0 + int(line['Geometry']['BoundingBox']['Height'] * concatenated_image_dims[1])
        
        # Check if y0 is within 5 of an existing key
        key = next((k for k in detected_lines if abs(k - y0) <= margin // 2), None)
        if key is None:
            
            # If it's not, add a new entry to the dictionary as y0
            detected_lines[y0] = {
                'text': line['Text'],
                'range': (y0, y1)
            }
        else:
            # If it is, append the new text to the existing text
            detected_lines[key]['text'] += ' ' + line['Text']
    
    # Iterate again over results
    for y0, line in detected_lines.items():
        
        min_y = line['range'][0]
        max_y = line['range'][1]

        # Iterate across metadata
        for key, value in list(crop_metadata.items())[metadata_index:]:

            # If the y coordinate is within the range, update the metadata text
            if check_in_range((min_y, max_y), value['range'], margin=margin):
                value['text'] += line['text']
                
                # Set metadata index to current index and break out of loop
                metadata_index = key
                break

    return crop_metadata

def textract_batched_inference(concatenated_image, concatenation_metadata, margin):
        
    # Convert to bytes
    image_bytes_io = io.BytesIO()
    concatenated_image.save(image_bytes_io, format='PNG')
    image_bytes = image_bytes_io.getvalue()

    try: 
        # Predict
        response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
        aligned_predictions = align_textract_results(response, concatenation_metadata, 
                                                     concatenated_image_dims=concatenated_image.size, 
                                                     margin=margin)
        
        # Restructure results and update outputs
        predictions = {}
        for key, value in aligned_predictions.items():
            predictions[value['image']] = [
                {
                    'Text': value['text']
                }
            ]
    
    except Exception as e:
        print("Error: ", e)
        predictions = {}
        for key, value in concatenation_metadata.items():
            predictions[value['image']] = [
                {
                    'Text': value['text']
                }
            ]        
    return predictions
      
def perform_batched_ocr_on_image(input_dir, dataset, model, evaluation_set, alphabet_set, output_dir="../outputs", subset=[], margin=10):
    
    # Set metadata
    timenow = str(datetime.now()).split('.')[0].replace('-','').replace(' ', '_').replace(':', '')
    output_filename = f"{output_dir}/{model}_inference_{timenow}_{dataset}_dataset_crops.json"
    metadata = {
        'input_dir': input_dir,
        'model': model,
        'dataset': dataset,
        'datetime': timenow
    }
    
    # Initialise predictions dict
    outputs = {
        'metadata': metadata,
        'predictions': {}
    }
    
    # Load all image names in the directory
    image_files = os.listdir(input_dir)
    
    if subset:
        image_files = [image for image in image_files if image in subset]
    
    # Iterate across evaluation set
    for evaluation_image_name in evaluation_set:
        
        for alphabet in alphabet_set:
            
            capture_name = f"{evaluation_image_name}_{alphabet}"
            
            if capture_name in subset:
                print(f"Skipping {capture_name}")
                continue
            
            # Select out relevant crops
            images = [image for image in image_files if capture_name in image]
            images.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
            
            
            # Concatenate vertically and get metadata
            concatenated_image, concatenation_metadata =  concatenate_images_vertically(images, input_dir, margin=margin)
            
            print(f"{str(datetime.now()).split('.')[0].replace('-','').replace(' ', '_').replace(':', '')}\tSize for image {capture_name}:\t{concatenated_image.size}")
            
            if model == 'document_intelligence':
                predictions = document_intelligence_batched_inference(concatenated_image, concatenation_metadata, margin=margin)
            else:
                predictions = textract_batched_inference(concatenated_image, concatenation_metadata, margin=margin)
            
            outputs['predictions'].update(predictions)
            
    # Output to output directory as json
    dump_dict_to_json(outputs, output_filename)
    print("Results stored at ", output_filename)
    return outputs

def tesseract_inference(image_path):
    
    # Read image file as binary data
    img = Image.open(image_path)

    # Infer text with Tesseract    
    text = pytesseract.image_to_string(img).strip()
    return [{"Text": text}]

def easyocr_inference(image_path):
    
    # Infer and process text with EasyOCR
    detections = easyocr_reader.readtext(image_path)
    text = " ".join([detection[1] for detection in detections])
    return [{"Text": text}]

def azure_document_intelligence_inference(image_path):
    
    # Read image file as binary data
    image = pad_to_50(plt.imread(image_path))
    image = Image.fromarray(image)
    image_stream = io.BytesIO()
    image.save(image_stream, format='PNG')
    image_stream.seek(0)
    
    # Call Azure Document AI to analyze the document
    poller = form_recognizer_client.begin_recognize_content(image_stream)
    result = poller.result()

    # Extract detected text from the result
    detected_text = []
    for page in result:
        for line in page.lines:
            
            detected_text += [word.text for word in line.words]

    # Return the detected text
    return [{"Text": " ".join([word for word in detected_text])}]

def aws_textract_inference(image_path):
    
    # Read image file as binary data
    with open(image_path, 'rb') as file:
        image_bytes = file.read()
    
    # Call Textract API to detect text in the image
    response = textract_client.detect_document_text(Document={'Bytes': image_bytes})
    
    # Extract detected text from the response
    detected_text = []
    for block in response['Blocks']:
        # Only consider blocks of type 'LINE' or 'WORD'
        if block['BlockType'] in ['LINE']:
            detected_text.append(block)
        
    # Return the detected text
    return detected_text

def model_inference(image_path, model):
    
    if model == 'tesseract':
        return tesseract_inference(image_path)
    elif model == 'easyocr':
        return easyocr_inference(image_path)
    elif model == 'document_intelligence':
        return azure_document_intelligence_inference(image_path)
    elif model == 'textract':
        return aws_textract_inference(image_path)
    else:
        raise ValueError(f"Model {model} not supported")

# Function to look within a directory and get all images and pass them all through a model and output resultant text to output folder
def perform_ocr_on_image(input_dir, dataset, model, output_dir="../outputs", subset=None):
    
    # Set metadata
    timenow = str(datetime.now()).split('.')[0].replace('-','').replace(' ', '_').replace(':', '')
    output_filename = f"{output_dir}/{model}_inference_{timenow}_{dataset}_dataset_crops.json"
    metadata = {
        'input_dir': input_dir,
        'model': model,
        'dataset': dataset,
        'datetime': timenow
    }
    
    # Initialise predictions dict
    outputs = {
        'metadata': metadata,
        'predictions': {}
    }
    
    # Load all image names in the directory
    image_files = os.listdir(input_dir)
    
    if subset:
        image_files = [image for image in image_files if image in subset]
    
    # Iterate across image names
    for image in image_files:
    
        # Call model inference
        try:
            detected_text = model_inference(f"{input_dir}/{image}", model)
        except Exception as e:
            print(f"Text detection somehow failing on {input_dir}/{image}\n{e}")
            detected_text = [{
                "Text": "",
                "Status": f"Text not found or errored out {e}"
            }]
        
        # Update dictionary
        outputs['predictions'][image] = detected_text
        
    # Output to output directory as json
    dump_dict_to_json(outputs, output_filename)
    print("Results stored at ", output_filename)
    return outputs

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Perform batched OCR on images')
    parser.add_argument('--input_dir', type=str, help='Input directory containing images')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name', choices=['document_intelligence', 'textract'])
    parser.add_argument('--method', type=str, help='Inference method - individual or batched', choices=['individual', 'batched'])
    args = parser.parse_args()
    
    # Azure
    if args.model == "document_intelligence":    
        load_dotenv()
        endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")
        form_recognizer_client = FormRecognizerClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    
    # AWS
    elif args.model == "textract":
        textract_client = boto3.client('textract')
    
    # EasyOCR
    elif args.model == "easyocr":
        easyocr_reader = easyocr.Reader(['en'])
    
    # Perofrm OCR
    if args.method == 'individual':
        results = perform_ocr_on_image(input_dir=args.input_dir, dataset=args.dataset, model=args.model)
    else:
        results = perform_batched_ocr_on_image(input_dir=args.input_dir, 
                                               dataset=args.dataset, 
                                               model=args.model, 
                                               evaluation_set=FUNSD_EVAL_SET, 
                                               alphabet_set=ALPHABET_SET)

