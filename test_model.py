"""
Documentation at https://github.com/dmacneill/AxisID. 
"""

import os
import re
import argparse
import torch
from PIL import Image
import numpy as np
import shared_functions as f
from model import Network

def crop_and_scale(image, target_size):
    
    """Crop and scale an image to a target_size by target_size square. First the
    larger axis is cropped to the size of the smaller axis, and then the image
    is re-sized to target_size.
    """
    
    size = image.size
    crop_axis = np.argmax(size)
    same_axis = np.argmin(size)
    to_crop = size[crop_axis] - size[same_axis]
    half_crop = to_crop//2
    
    top =  0 if same_axis == 1 else half_crop
    bottom = size[1] if same_axis == 1 else top+size[0]
    left = 0 if same_axis == 0 else half_crop
    right = size[0] if same_axis == 0 else left+size[1]
    
    cropped_image = image.crop((left, top, right, bottom))
    scaled_image = cropped_image.resize((target_size, target_size), Image.ANTIALIAS)
    
    return scaled_image

def read_annotation_file(annotation_dir, basename):
    
    """Reads the annoation file for the image with name (after dropping the extension) 
    basename. The annotation files are assumed to be in the format output by the
    axis labeling tool available at https://github.com/dmacneill/axis-annotation-tool.
    """
    
    annotation_path = annotation_dir+'/'+basename+'-edges.csv'
    
    try:
        with open(annotation_path) as annotation_file:
            header = annotation_file.readline()
            angle = float(re.search('\d+\.\d+',header)[0])
            return angle
    except FileNotFoundError:
        print('Missing annotation file: ', annotation_path)
        
def read_results_file(results_path):
    
    """Reads the results file saved by test_model
    Args:
        results_path: path to the results file
    Returns:
        list of filenames, list of angle predictions, list of ground-truth angles
    """
    
    with open(results_path) as results_file:
        results_file.readline()
        rows = results_file.read().splitlines()
        names, predictions, values = list(zip(*[row.split(',') for row in rows]))
        predictions =  list(map(float, predictions))
        values = list(map(float, values))
    
    return names, predictions, values

def dot_loss(yhat, y):
    """Cosine similarity loss function
    """
    return np.mean(1-np.sum(yhat*y, axis = -1))

def test_model(model, params):
    
    """Use the model to predict axis angles for all images in a directory. The images
    are cropped and re-sized to the model's input size before prediction. The path to
    a directory of angle annotations for the images can optionally be passed, in which
    case the loss between predicted and ground-truth values is computed.
    Args:
        model: nn.Module
        params: dict of params
    """
    
    image_dir = params['Image Directory']
    annotation_dir = params['Annotations Directory']
    target_size = params['Target Size']
    device = params['Device']
    
    images = os.listdir(image_dir)
    images = [image for image in images if image.endswith('.tif') or image.endswith('.jpg')]
    
    vector_predictions = []
    true_angles = []
    
    for image_path in images:
        
        print('Processing image: ', image_path)
        
        image = Image.open(image_dir+'/'+image_path)
        image = crop_and_scale(image, target_size)
        image = np.array(image)
        image = image.transpose((2,0,1))
        
        x = torch.from_numpy(image)
        x = x.unsqueeze(dim = 0)
        x = x.to(device)
        
        with torch.no_grad():
            prediction = model(x).to('cpu').numpy()
            vector_predictions.append(prediction[0])
        
        true_angle = read_annotation_file(annotation_dir, image_path.split('.')[0].split('-')[0])
        true_angles.append(true_angle)
        
    vector_predictions = np.array(vector_predictions)
    angle_predictions = f.vectoangle(vector_predictions)
    
    true_angles = np.array(true_angles)
    true_vectors = f.angletovec(true_angles)
    loss = dot_loss(true_vectors, vector_predictions)
    errors = f.circular_difference(true_angles, angle_predictions)
    rms_errors = np.sqrt(np.mean(errors**2))
    print('Loss: ', '{:.2f}'.format(loss))
    print('RMS error: ', '{:.2f}'.format(rms_errors), 'Degrees')
        
    with open('test_results.csv', 'w+') as output_file:
        output_file.write('# loss: '+'{:.2f}'.format(loss)+', RMS error: '+'{:.1f}'.format(rms_errors)+' Degrees\n')
        for i, image in enumerate(images):
            output_file.write(image+','+'{:.2f}'.format(angle_predictions[i])+','+'{:.1f}'.format(true_angles[i])+'\n')   
                
def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = '@')
    
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use GPU if available')
    
    parser.add_argument('--image_dir', type = str, default = 'images',
                        help = 'test image directory')
    
    parser.add_argument('--annotation_dir', type = str, default = 'annotations',
                        help = 'annotation file directory')
    
    parser.add_argument('--weights_path', type = str, default = 'output/model_weights-final.pth',
                        help = 'path to model weights')
    
    parser.add_argument('--target_size', type = int, default = 256,
                        help = 'size of images input to model')
    
    args = parser.parse_args()   
    
    params = dict()
    
    params['Image Directory'] = args.image_dir
    params['Annotations Directory'] = args.annotation_dir
    params['Target Size'] = args.target_size
    params['Device'] = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    
    model = Network()
    model.load_state_dict(torch.load(args.weights_path))
    model.to(params['Device'])
    model.eval()
    
    test_model(model, params)
    
if __name__ == '__main__':
    main()