      
#####################################################################################################
                                            # Nuevo modelo
#####################################################################################################

import argparse
import json
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.utils.data
import boto3
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets
from io import BytesIO

# imports the model in model.py by name
from model import Net


def model_fn(model_dir):
    """Load the PyTorch model from the `model_dir` directory."""
    print("Loading model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net(model_info['input_features'], model_info['hidden_dim'], model_info['output_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    model.to(device).eval()

    print("Done loading model.")
    return model


# Gets training data in batches from the train.csv file
def _get_train_data_loader(batch_size, training_dir, test_dir):
    
    print("Get train data loader.")
    num_workers = 1
    train_dir = os.path.join(training_dir)
    test_dir =  os.path.join(test_dir)
    
    data_transforms = {'train':transforms.Compose([transforms.Resize(258),
                                                   transforms.RandomResizedCrop(224),
                                                   transforms.RandomRotation(30),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                       'test':transforms.Compose([transforms.Resize(258),
                                                  transforms.CenterCrop(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]),
                      }
    train_data = datasets.ImageFolder(train_dir, transform = data_transforms['train'])
    test_data = datasets.ImageFolder(test_dir, transform = data_transforms['test'])
        
    trainloader = torch.utils.data.DataLoader(train_data, batch_size = batch_size, num_workers = num_workers, shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = batch_size, num_workers = num_workers, shuffle = False)
    

    return trainloader, testloader


# Provided training function
def train(n_epochs, loaders, model, optimizer, criterion, valid_loader):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 

    print_every = 5

    train_losses, test_losses = [], []
    for epoch in range(0, n_epochs):
        
#         initialize variables to monitor training and validation loss
        train_loss = 0
        model.train()
        total =0 
        print('Epoch: {}'.format(epoch))
        for index, (inputs, labels) in enumerate(loaders):               
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad() 
            
            output = model(inputs) 
            
            loss = criterion(output, labels)          
            loss.backward()
            optimizer.step()
#             train_loss += loss.item()
            train_loss = train_loss + ((1 / (index + 1)) * (loss.data - train_loss))
            total += inputs.size(0)
            
        print('Train Loss: {:.6f}\n'.format(train_loss))

        if epoch % print_every == 0:
            
            correct = 0
            total = 0
            model.eval()
            valid_loss = 0
            for batch_idx, (data, target) in enumerate(valid_loader):

                inputs = data.to(device)
                target = target.to(device)
                    
                output_v = model(inputs)
                loss = criterion(output_v, target)
#                 valid_loss += loss.item()
                    
                # convert output probabilities to predicted class
                pred = output_v.data.max(1, keepdim=True)[1]
                # compare predictions to true label
                
                correct += np.sum(np.squeeze(pred.eq(target.data.view_as(pred))).cpu().numpy())
                total += data.size(0)
                
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
            
            print('\nTest Accuracy: %2d%% (%2d/%2d)' % (100. * correct / total, correct, total))
            print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
                train_loss,
                valid_loss
            ))
                                      
        model.train()


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()
    
    
    
    parser.add_argument('--hosts', type=list, default=json.loads(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 10)')
    
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0001)')
    
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    ## TODO: Add args for the three model parameters: input_features, hidden_dim, output_dim
    # Model Parameters
    parser.add_argument('--input_features', type=int, default=2, metavar='IN')
    parser.add_argument('--hidden_dim', type=int, default=10, metavar='H')
    parser.add_argument('--output_dim', type=int, default=1, metavar='OUT')
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load the training data.
    train_loader, test_loader = _get_train_data_loader(args.batch_size, args.data_dir, args.test)
    


    ## --- Your code here --- ##
    
    ## TODO:  Build the model by passing in the input params
    # To get params from the parser, call args.argument_name, ex. args.epochs or ards.hidden_dim
    # Don't forget to move your model .to(device) to move to GPU , if appropriate
    model = Net(args.input_features, args.hidden_dim, args.output_dim).to(device)

    ## TODO: Define an optimizer and loss function for training
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Trains the model (given line of code, which calls the above training function)
    train(args.epochs, train_loader, model, optimizer, criterion, test_loader)


    ## TODO: complete in the model_info by adding three argument names, the first is given
    # Keep the keys of this dictionary as they are 
    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'input_features': args.input_features,
            'hidden_dim': args.hidden_dim,
            'output_dim': args.output_dim
        }
        torch.save(model_info, f)
        
    ## --- End of your code  --- ##
    

	# Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)   
    
    
    
    
    