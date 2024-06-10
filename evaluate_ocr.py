import os
import json
import argparse
from utils.common import dump_dict_to_json, get_timenow
from torchmetrics.text import WordErrorRate, CharErrorRate, EditDistance

def evaluate_ocr_model(annotation_dir, inference_outputs_filepath, evaluation_objective, model, output_dir="../outputs", save=True):
    
    # Load inference outputs
    inference_outputs = json.load(open(inference_outputs_filepath))
    
    # Get all annotations
    annotations = os.listdir(annotation_dir)
    
    # Instantiate lists and scoring dict
    all_predictions = []
    all_annotations = []
    timenow = get_timenow()
    
    scoring = {
        model: {
            'metadata': {
                'inference_metadata': inference_outputs['metadata'],
                'annotation_dir': annotation_dir,
                'inference_outputs_filepath': inference_outputs_filepath,
                'evaluation_objective': evaluation_objective,
                'datetime': timenow,
            },
            'evaluation_pairs': {}
        }
    }
    
    # Iterate across annotations
    for annotation in annotations:
    
        # Load annotation txt file
        with open(os.path.join(annotation_dir, annotation)) as f:
            annotation_text = f.read()
    
        # Process annotation name
        image_name = annotation.split('.')[0]
        
        # Get corresponding prediction
        predicted_lines = inference_outputs['predictions'][f'{image_name}.png']
        predicted_text = ''
        
        for line in predicted_lines:
            predicted_text += line["Text"]
            
        scoring[model]['evaluation_pairs'][image_name] = {
            'annotation': annotation_text,
            'prediction': predicted_text,
        }
        all_annotations.append(annotation_text)
        all_predictions.append(predicted_text)
        
    # Score
    wer = WordErrorRate()
    cer = CharErrorRate()
    edit_distance = EditDistance()
    wer_score = wer(preds=all_predictions, target=all_annotations)
    cer_score = cer(preds=all_predictions, target=all_annotations)
    edit_distance_score = edit_distance(preds=all_predictions, target=all_annotations)
    
    # Update scoring dict
    scoring[model]['scores'] = {
        'word_error_rate': wer_score.item(),
        'char_error_rate': cer_score.item(),
        'edit_distance': edit_distance_score.item()
    }
    
    # Output to output directory as json
    if save:
        output_filename = f"{output_dir}/ocr_evaluation_{timenow}.json"
        dump_dict_to_json(scoring, output_filename)
        print("Results stored at ", output_filename)
    return scoring

def evaluate_all_models(annotation_dir, inference_outputs_directory, evaluation_objective, output_dir="../outputs"):
    
    all_scores = {}
    inference_outputs = os.listdir(inference_outputs_directory)
    
    for inference_output in inference_outputs:
        
        model = inference_output.split('_inference')[0]
        inference_outputs_filepath = f"{inference_outputs_directory}/{inference_output}"
        
        print(f"Evaluating model: {model}")
        all_scores[model] = evaluate_ocr_model(annotation_dir=annotation_dir, 
                                               inference_outputs_filepath=inference_outputs_filepath, 
                                               evaluation_objective=evaluation_objective, 
                                               model=model, 
                                               output_dir=output_dir, 
                                               save=False)
    # Output to JSON
    timenow = get_timenow()
    output_filename = f"{output_dir}/{evaluation_objective}_{timenow}.json"
    dump_dict_to_json(all_scores, output_filename)
    return all_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate OCR models in terms of recognition performance')
    parser.add_argument('--annotation_dir', '-a', type=str, required=True, help='Path to annotations directory')
    parser.add_argument('--inference_outputs_directory', '-i', type=str, required=True, help='Path to inference outputs directory')
    parser.add_argument('--evaluation_objective', '-e', type=str, required=True, help='Objective of evaluation')
    parser.add_argument('--output_dir', '-o', type=str, help='Path to output directory')
    args = parser.parse_args()
    
    evaluate_all_models(annotation_dir=args.annotation_dir, 
                        inference_outputs_directory=args.inference_outputs_directory, 
                        evaluation_objective=args.evaluation_objective, 
                        output_dir=args.output_dir)