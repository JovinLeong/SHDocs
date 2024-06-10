import os
import cv2
import torch
import argparse
import numpy as np
from utils.common import dump_dict_to_json, get_timenow
from torchmetrics.image import UniversalImageQualityIndex, StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def assess_greyscale_image_quality(ground_truth_image, enhanced_image, psnr, ssim, uiqi):
    
    # Unsqueeze to add batch and channel dimensions
    enhanced_image = enhanced_image.unsqueeze(0).unsqueeze(0)
    ground_truth_image = ground_truth_image.unsqueeze(0).unsqueeze(0)
    
    return {
        'psnr': psnr(enhanced_image, ground_truth_image).item(),
        'ssim': ssim(enhanced_image, ground_truth_image).item(),
        'uiqi': uiqi(enhanced_image, ground_truth_image).item(),
    }

def evaluate_greyscale_image_quality(ground_truth_image_directory, enhanced_image_directory, evaluation_objective, model, output_dir="../outputs"):
    
    timenow = get_timenow()
    output_filename = f"{output_dir}/{model}_image_quality_evaluation_{timenow}.json"
    
    scoring = {
        model: {
            'metadata': {
                'enhanced_image_directory': enhanced_image_directory,
                'ground_truth_image_directory': ground_truth_image_directory,
                'evaluation_objective': evaluation_objective,
                'datetime': timenow,
            },
            'evaluation_pairs': {}
        }
    }
    
    # Instantiate all images and metrics
    all_images = os.listdir(ground_truth_image_directory)
    all_images = [image for image in all_images if image[0] != '.'] 
    psnr = PeakSignalNoiseRatio()
    ssim = StructuralSimilarityIndexMeasure()
    uiqi = UniversalImageQualityIndex()
    
    for image in all_images:
        
        # Load images as tensors
        ground_truth_image = cv2.imread(os.path.join(ground_truth_image_directory, f'{image}'), cv2.IMREAD_GRAYSCALE)
        specular_highlight_image = cv2.imread(os.path.join(enhanced_image_directory, f'{image}'), cv2.IMREAD_GRAYSCALE)
        ground_truth_image = torch.Tensor(ground_truth_image)
        specular_highlight_image = torch.Tensor(specular_highlight_image)
        
        # Assess image quality
        scoring[model]['evaluation_pairs'][image[:-4]] = assess_greyscale_image_quality(ground_truth_image, specular_highlight_image, psnr, ssim, uiqi)
    
    # Get values as a n x 3 matrix corresponding to the three metrics
    scores = scoring[model]['evaluation_pairs'].values()
    psnr_scores = []
    ssim_scores = []
    uiqi_scores = []
    
    for score in scores:
        psnr_scores.append(score['psnr'])
        ssim_scores.append(score['ssim'])
        uiqi_scores.append(score['uiqi'])
    
    # Update scoring dict
    scoring[model]['scores'] = {
        'psnr': np.mean(psnr_scores),
        'ssim': np.mean(ssim_scores),
        'uiqi': np.mean(uiqi_scores)
    }
    
    # Output to output directory as json
    dump_dict_to_json(scoring, output_filename)
    print("Results stored at ", output_filename)
    return scoring

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate image quality')
    parser.add_argument('--ground_truth_image_directory', '-g', type=str, help='Path to ground truth images', required=True)
    parser.add_argument('--enhanced_image_directory', '-e', type=str, help='Path to enhanced images', required=True)
    parser.add_argument('--evaluation_objective', '-o', type=str, help='Objective of the evaluation', required=True)
    parser.add_argument('--model', '-m', type=str, help='Model being evaluated', required=True)
    parser.add_argument('--output_dir', '-d', type=str, help='Output directory for evaluation results', default="../data/outputs/image_quality_evaluations")
    parser.add_argument('--output_dir', '-d', type=str, help='Output directory for evaluation results', default="../data/outputs/image_quality_evaluations")
    args = parser.parse_args()

    evaluate_greyscale_image_quality(ground_truth_image_directory=args.ground_truth_image_directory,
                                     enhanced_image_directory=args.enhanced_image_directory,
                                     evaluation_objective=args.evaluation_objective, 
                                     model=args.model, 
                                     output_dir=args.output_dir)
