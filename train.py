"""
Documentation at https://github.com/dmacneill/AxisID. 
"""

import os
import random
from PIL import Image
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import Network
import schedulers
import shared_functions as f

class AngleDataset(Dataset):
    
    """torch.utils.data.Dataset subclass for holding image data and 
    corresponding angle labels.
    Attributes:
        X: torch.tensor of images with shape (num_samples, num_channels, height, width)
        y: torch.tensor of angle labels in vector format, with shape (num_samples, 2)
    """
    
    def __init__(self, X, y, transform = None):
        self.X = X
        self.y = y
        self.transform = transform
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        
        sample = self.transform((self.X[idx], self.y[idx]))
                                       
        return sample

class FourfoldRotation():
    
    """Rotates image by 0, 90, 180 or 270 degrees, with equal probability. Depending
    on the value of shared_functions.modulus the label could be invariant under
    these operations or require a correction (for values of 30 or 90, the labels
    are invariant).
    """
    
    def __call__(self, sample):
        
        X, y = sample
        k = int(torch.randint(0,4,(1,)))
        
        theta_r = (f.k*k*90)%360
        c = np.cos(np.pi*theta_r/180)
        s = np.sin(np.pi*theta_r/180)
        y = torch.matmul(y, torch.tensor([[c,s],[-s,c]], dtype = torch.float))
        
        return (torch.rot90(X, k = k, dims = [-2,-1]), y)
    
class VerticalFlip():
    
    """Flips the image vertically or doesn't, with equal probability. For any
    value of the modulus, this operation flips the second label component.
    """
    
    def __call__(self, sample):
        
        X, y = sample 
        k = int(torch.randint(0,2,(1,)))
        
        if k == 1:
            return (torch.flip(X, dims = [-2]), y*torch.tensor([1.0,-1.0]))
        else:
            return sample

def read_angles_file(angles_path):
    
    """Reads the angles file containing the ground-truth angle assignments. This file contains
    rows of entries in the form: basename,angle,count. Basename is a name assigned to each image used to 
    generate the training examples, and count is the number of times images derived from this image
    through pre-computed data augmentation appear in the training images.
    Args:
        angles_path: path to the angles file
    Returns:
        list of basenames, list of angle labels, list of counts
    """
    
    with open(angles_path) as angles_file:
        rows = angles_file.read().splitlines()
        basenames, angles, counts = list(zip(*[row.split(',') for row in rows]))
        angles =  list(map(float, angles))
        counts = list(map(int, counts))
    
    return basenames, angles, counts

def load_images(idx, basenames, angles, counts, image_dir):
    
    """Loads training images and prepares the tensors required for an AngleDataset
    Args:
        idx: list of indices of basenames to use for the dataset
        basenames: list of all basenames
        angles: list of all angle labels
        counts: list of all counts
        image_dir: directory of training images
    Returns: X, y; X is a torch.tensor of shape (num_samples, num_channels, height, width) and y is a tensor
    of label vectors with shape (num_samples, 2)
    """
    
    images = os.listdir(image_dir)
    images = [image for image in images if image.endswith('.tif') or image.endswith('.jpg')]
    extensions = [(image.split('-')[0], image.split('.')[1]) for image in images]
    extensions = dict(extensions)
    
    X = []
    y = []
    
    for i in idx:
        
        basename = basenames[i]
        angle = angles[i]
        count = counts[i]
        extension = extensions[basename]

        for j in range(count):#Loop over all versions of an image generated through pre-computed data augmentation
            try:
                image = Image.open(image_dir+'/'+basename+'-'+str(j)+'.'+extension)
                image = np.array(image)
                image = image.transpose((2,0,1))
                X.append(image)
                y.append(angle)
            except FileNotFoundError:
                print('Missing image file', basename+'-'+str(j)+'.'+extension)
    
    X = np.array(X)
    X = torch.from_numpy(X)
    y = np.array(y)
    y = f.angletovec(y)#labels are converted to a vector format
    y = torch.tensor(y, dtype = torch.float)
     
    return X, y
   
def create_datasets(angles_path, image_dir, f_split, batch_size, num_workers):
    
    """Creates DataSets and DataLoaders for the training and validation sets
    """
    
    basenames, angles, counts = read_angles_file(angles_path)
    
    N_samples = len(basenames)
    idx = random.sample(range(N_samples), N_samples)
    N_train = np.floor(f_split*N_samples).astype(dtype = 'int')
    data_transforms = transforms.Compose([FourfoldRotation(), VerticalFlip()]) 
    
    print('Loading training data')
    X, y = load_images(idx[0:N_train], basenames, angles, counts, image_dir)   
    train_dataset = AngleDataset(X,y, transform = data_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, 
                              shuffle = True, pin_memory = True, num_workers = num_workers)
    print('Loading validation data')
    X, y = load_images(idx[N_train:], basenames, angles, counts, image_dir)
    val_dataset = AngleDataset(X,y, transform = data_transforms)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, 
                            shuffle = True, pin_memory = True)
    val_names = [basenames[i] for i in idx[N_train:]]
    with open('validation_examples.txt', 'w+') as val_names_file:
        for item in val_names:
            val_names_file.write(item+'\n')
    
    return train_dataset, train_loader, val_dataset, val_loader

def calculate_errors(model, device, X, y):
    
    """Calculates the angular errors in model predictions
    Args:
        model: nn.Module to test
        device: torch.device to carry out calculation
        X: torch.tensor containing image data
        y: ground-truth angle labels
    Returns:
        ndarray of angle errors
    """
    
    predictions = []

    model.eval()

    with torch.no_grad():

        for x in X:
            
            x = x.to(device)
            x = x.unsqueeze(dim = 0)
            prediction = f.vectoangle(model(x).to('cpu').numpy())
            predictions.append(prediction[0])
        
    predictions = np.array(predictions)
    values = f.vectoangle(y.numpy())
    
    errors = f.circular_difference(predictions, values)
    
    return errors

def dot_loss(yhat, y):
    
    """Cosine similarity loss function
    """
    
    return torch.mean(1-torch.sum(yhat*y, axis = -1))

def train(model, optimizer, scheduler, train_loader, val_loader, params):
    
    """Trains the model
    
    Args:
        model: nn.Module
        optimizer: torch.optim.Optimizer
        scheduler: learning rate scheduler with .step() method
        train_loader: DataLoader for training set
        val_loader: DataLoader for validation set
        params: dict of params
    Returns:
        ndarray of training losses, ndarray of validation losses
    """
    
    print('Starting training')
    device = params['Device']
    epochs = params['Epochs']
    output_frequency = params['Output Frequency']
    save_frequency = params['Save Frequency']
    output_dir = params['Output Directory']
    
    losses = []
    val_losses = []

    for epoch in range(epochs):
    
        epoch_losses = []
        epoch_val_losses = []
        
        model.train()
    
        for x_batch, y_batch in train_loader:
        
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            loss = dot_loss(y_batch, model(x_batch))
        
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.detach().to('cpu').numpy())
        
        epoch_loss = np.mean(epoch_losses)
        losses.append(epoch_loss)
    
        if (epoch+1)%output_frequency == 0:
            print('Training loss on epoch '+str(epoch+1)+': '+ '{:.2f}'.format(epoch_loss))
    
        with torch.no_grad():
        
            model.eval()
        
            for x_val, y_val in val_loader:
            
                x_val = x_val.to(device)
                y_val = y_val.to(device)
                val_loss = dot_loss(y_val, model(x_val))
                epoch_val_losses.append(val_loss.to('cpu').numpy())
            
            epoch_val_loss = np.mean(np.array(epoch_val_losses))
            val_losses.append(epoch_val_loss)
        
            if (epoch+1)%output_frequency == 0:
                print('Validation loss on epoch '+str(epoch+1)+': '+ '{:.2f}'.format(epoch_val_loss))
        
            if (epoch+1)%save_frequency == 0:
                results = np.column_stack((np.arange(1,epoch+2), np.array(losses), np.array(val_losses)))
                np.savetxt(output_dir+'/losses.csv', results, delimiter = ',')
                torch.save(model.state_dict(), output_dir+'/'+'model_weights-'+str(epoch+1)+'.pth')
        
        if scheduler is not None:
            scheduler.step()
        
    return np.array(losses), np.array(val_losses)

def main():
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, fromfile_prefix_chars = '@')
    
    parser.add_argument('--cuda', action = 'store_true', 
                        help = 'use GPU if available')
    
    parser.add_argument('--weight_decay', type = float, default = 0.05,
                        help = 'weight decay')
    
    parser.add_argument('--lr', type = float, default = 3e-4,
                        help = 'learning rate')
    
    parser.add_argument('--momentum', type = float, default = 0.9,
                        help = 'momentum')
    
    parser.add_argument('--optimizer', default = 'Adam', choices = ['Adam', 'SGD'],
                        help = 'optimizer type')
    
    parser.add_argument('--scheduler', default = None, choices = ['OneCycle', 'StepDecay'],
                        help = 'scheduler type')
    
    parser.add_argument('--epochs', type = int, default = 2000, 
                        help = 'training epochs')
    
    parser.add_argument('--output_frequency', type = int, default = 10, 
                        help = 'epochs between loss output')
    
    parser.add_argument('--save_frequency', type = int, default = 100, 
                        help = 'epochs between saving weights')
    
    parser.add_argument('--output_dir', type = str, default = 'output', 
                        help = 'output directory')
    
    parser.add_argument('--image_dir', type = str, default = 'images',
                        help = 'training image directory')
    
    parser.add_argument('--angles_path', type = str, default = 'angles.csv',
                        help = 'path to labels file')
    
    parser.add_argument('--weights_path', type = str, default = None,
                        help = 'path to initial weights')
    
    parser.add_argument('--f_split', type = float, default = 0.8, 
                        help = 'fraction of examples to use for training set')
    
    parser.add_argument('--batch_size', type = int, default = 64, 
                        help = 'batch size')
    
    parser.add_argument('--num_workers', type = int, default = 0, 
                        help = 'Number of workers for DataLoader')
    
    parser.add_argument('--scheduler_params', type = float, nargs = '*', default = None,
                        help = 'parameters passed to learning rate scheduler')
    
    parser.add_argument('--seed', type = int, default = 2319,
                        help = 'seed for random number generators')
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    #torch.backends.cudnn.benchmark = False
    #torch.use_deterministic_algorithms(True)
    
    train_dataset, train_loader, val_dataset, val_loader = create_datasets(args.angles_path, 
                                                                           args.image_dir, args.f_split, 
                                                                           args.batch_size, args.num_workers)
    
    params = dict()
    params['Device'] = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    params['Epochs'] = args.epochs
    params['Output Frequency'] = args.output_frequency
    params['Save Frequency'] = args.save_frequency
    params['Output Directory'] = args.output_dir
    
    model = Network()
    model.to(params['Device'])
    
    if args.weights_path:
        model.load_state_dict(torch.load(args.weights_path))
    
    if args.optimizer == 'Adam':
        optimizer = torch.optim.AdamW(model.parameters(), betas = (args.momentum, 0.99), lr = args.lr, weight_decay = args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr = args.lr, momentum = args.momentum, weight_decay = args.weight_decay)
    
    if args.scheduler == 'OneCycle':
        scheduler = schedulers.OneCycle(optimizer, args.scheduler_params)
        print(optimizer.param_groups[0]['lr'])
    elif args.scheduler == 'StepDecay':
        scheduler = schedulers.StepDecay(optimizer, args.scheduler_params)
    else:
        scheduler = None
    
    losses, val_losses = train(model, optimizer, scheduler, train_loader, val_loader, params)
    
    torch.save(model.state_dict(), args.output_dir+'/model_weights-final.pth')
    
    results = np.column_stack((np.arange(1,args.epochs+1), losses, val_losses))
    np.savetxt(args.output_dir+'/losses.csv', results, delimiter = ',')
    
    plt.figure()
    plt.plot(np.arange(1,args.epochs+1), np.log10(val_losses))
    plt.plot(np.arange(1,args.epochs+1), np.log10(losses))
    plt.xlim([1, args.epochs+1])
    plt.xticks(fontsize = 14);
    plt.yticks(fontsize = 14);
    plt.ylabel('log$_{10}$(Loss)', fontsize = 15);
    plt.xlabel('Epoch', fontsize = 15);
    plt.legend(['Validation', 'Training'], fontsize = 15, loc = 'best')
    plt.tight_layout()
    plt.savefig(args.output_dir+'/training_losses.pdf')
    plt.close()
    
    errors = calculate_errors(model, params['Device'], val_dataset.X, val_dataset.y)
    rms_error = np.sqrt(np.mean(errors**2))
    print('RMS angle error of final model:', '{:.2f}'.format(rms_error))
    
    plt.figure()
    plt.hist(np.abs(errors), log = True, density = True);
    plt.plot((rms_error, rms_error),(1e-4,1),'--k')
    plt.text(rms_error, 0.5, ' RMS error = '+'{:.1f}'.format(rms_error)+' Degrees', fontsize = 12)
    plt.ylim([1e-4, 1])
    plt.xlim(0,f.modulus//2)
    plt.xticks(fontsize = 14);
    plt.yticks(fontsize = 14);
    plt.ylabel('Probability', fontsize = 15);
    plt.xlabel('Error Magnitude (Degrees)', fontsize = 15);
    plt.tight_layout()
    plt.savefig(args.output_dir+'/final_validation_errors.pdf')
    plt.close()
    
    print('Finished training')
    
if __name__ == '__main__':
    main()