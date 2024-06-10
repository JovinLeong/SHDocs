# Based on https://github.com/OCRComparison/dataset by Ricciuti Federico

import os
from PIL import Image
import json
import numpy as np
import argparse
from tqdm import tqdm
from constants import ALPHABET_MAPPING, ALPHABET_SET

def load_dataset(images_path, annotations_path, bbox_buffer, dataset):
    
    instances = [] 
    
    for annotation_file in os.listdir(annotations_path):
        
        # Determine how to access the data
        if dataset in ['s0_norm', 'deglared']:
            
            # Identify image directories
            image_files = [os.path.join(images_path, f"{annotation_file.replace('.json','')}_{alphabet}", f"{dataset}.png") for alphabet in ALPHABET_MAPPING.values()]
            
            # Create image paths
            for image_file in image_files:
                with Image.open(image_file) as image:
                    image = np.array(image)
                annotation = json.load(open(os.path.join(annotations_path, annotation_file), encoding='utf-8'))
                form = annotation['form']
                for idx in range(len(form)):
                    
                    filename = image_file.split('/')[-1].replace('.png','')+'_'+str(idx)
                    
                    element = form[idx]
                    x0, y0, x1, y1 = element['box']
                    x0 = max(0, x0 - bbox_buffer)
                    y0 = max(0, y0 - bbox_buffer)
                    x1 = min(image.shape[1], x1 + bbox_buffer)
                    y1 = min(image.shape[0], y1 + bbox_buffer)
                    subimage = image[y0: y1, x0: x1]
                    text = element['text']
                    instance = {
                        'image': subimage,
                        'text': text,
                        'filename': filename
                    }
                    instances.append(instance)
        
        elif dataset == 'camera':
            image_file = os.path.join(images_path, annotation_file.replace('.json',''), 's0_norm.png')

            with Image.open(image_file) as image:
                image = np.array(image)
            annotation = json.load(open(os.path.join(annotations_path, annotation_file), encoding='utf-8'))
            form = annotation['form']
            for idx in range(len(form)):
                element = form[idx]
                x0, y0, x1, y1 = element['box']
                x0 = max(0, x0 - bbox_buffer)
                y0 = max(0, y0 - bbox_buffer)
                x1 = min(image.shape[1], x1 + bbox_buffer)
                y1 = min(image.shape[0], y1 + bbox_buffer)
                subimage = image[y0: y1, x0: x1]
                text = element['text']
                instance = {
                    'image': subimage,
                    'text': text,
                    'filename': annotation_file.replace('.json','')+'_'+str(idx)
                }
                instances.append(instance)
           
        elif dataset == 'enhanced':
            
            for alphabet in ALPHABET_SET:
            
                image_file = os.path.join(images_path, f"{annotation_file.replace('.json','')}_{alphabet}.png")

                with Image.open(image_file) as image:
                    image = np.array(image)
                annotation = json.load(open(os.path.join(annotations_path, annotation_file), encoding='utf-8'))
                form = annotation['form']
                for idx in range(len(form)):
                    element = form[idx]
                    x0, y0, x1, y1 = element['box']
                    x0 = max(0, x0 - bbox_buffer)
                    y0 = max(0, y0 - bbox_buffer)
                    x1 = min(image.shape[1], x1 + bbox_buffer)
                    y1 = min(image.shape[0], y1 + bbox_buffer)
                    subimage = image[y0: y1, x0: x1]
                    text = element['text']
                    instance = {
                        'image': subimage,
                        'text': text,
                        'filename': image_file.split('/')[-1].replace('.png','')+'_'+str(idx)
                    }
                    instances.append(instance)
                            
        else:
            image_file = os.path.join(images_path, annotation_file.replace('.json','.png'))

            with Image.open(image_file) as image:
                image = np.array(image)
            annotation = json.load(open(os.path.join(annotations_path, annotation_file), encoding='utf-8'))
            form = annotation['form']
            for idx in range(len(form)):
                element = form[idx]
                x0, y0, x1, y1 = element['box']
                x0 = max(0, x0 - bbox_buffer)
                y0 = max(0, y0 - bbox_buffer)
                x1 = min(image.shape[1], x1 + bbox_buffer)
                y1 = min(image.shape[0], y1 + bbox_buffer)
                subimage = image[y0: y1, x0: x1]
                text = element['text']
                instance = {
                    'image': subimage,
                    'text': text,
                    'filename': annotation_file.replace('.json','')+'_'+str(idx)
                }
                instances.append(instance)
                    
    return instances

def save_extracted(instances, data_path, save_annotations=True):
    
    images_path = os.path.join(data_path, 'images')
    os.mkdir(images_path)
    
    if save_annotations:
        annotations_path = os.path.join(data_path, 'annotations')
        os.mkdir(annotations_path)
    
    for instance in tqdm(instances):
        
        image_file = os.path.join(images_path, instance['filename'] + '.png')

        image = Image.fromarray(instance['image'])
        text = instance['text']
        
        try:
            image.save(image_file)
        except:
            print(f"Error saving image: {image_file}")
            print(f"Instance: {instance}")
            print(f"Image: {image}")
            raise Exception(f"Error saving image: {image_file}")
            
        if save_annotations:
            annotation_file = os.path.join(annotations_path, instance['filename'] + '.txt')
            with open(annotation_file, 'w', encoding='utf-8') as f:
                f.write(text)
                f.close()
            
    return None

if(__name__=='__main__'):
    
    parser = argparse.ArgumentParser(description='Process FUNSD dataset to extract the images and annotations')
    parser.add_argument('--images_path', '-i', type=str, required=True, help='Path to images')
    parser.add_argument('--annotations_path', '-a', type=str, required=True, help='Path to annotations')
    parser.add_argument('--output_path', '-o', type=str, required=True, help='Path to outputs')
    parser.add_argument('--dataset', '-d', type=str, required=False, default='funsd', help='Type of dataset: funsd, s0_norm, deglared, camera')
    parser.add_argument('--bbox_buffer', '-b', type=int, required=False, help='Buffer for bounding box', default=1)
    parser.add_argument('--save_annotations', '-s', type=bool, required=False, help='Buffer for bounding box', default=True)
    args = parser.parse_args()

    instances = load_dataset(args.images_path, args.annotations_path, args.bbox_buffer, args.dataset)
    save_extracted(instances, args.output_path)