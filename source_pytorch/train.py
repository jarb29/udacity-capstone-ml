      
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
    
    train_transform = transforms.Compose([ transforms.Resize(224),
                                           transforms.RandomHorizontalFlip(), # randomly flip and rotate
                                           transforms.RandomRotation(10),
                                           transforms.CenterCrop(244),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                                         ])
    
    test_transform = transforms.Compose([
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.CenterCrop(244),
                                         transforms.Normalize([0.5, 0.5, 0.5],[0.5, 0.5, 0.5])
                                        ])
    
    
    train_data = datasets.ImageFolder(training_dir, transform=train_transform)

    test_data = datasets.ImageFolder(test_dir, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

    test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)

    return train_loader, test_loader


# Provided training function
def train(n_epochs, loaders, model, optimizer, criterion, valid_loader):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 

    #print_every = 10

    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        model.train()
        for index, (inputs, labels) in enumerate(loaders):
            # Move input and label tensors to the default device
            
            optimizer.zero_grad()
            output = model(inputs)
            print(labels, "ANTES LAS ETIQUETAS")
            labels = labels.type(torch.FloatTensor)
            print(labels, "LAS ETIQUETAS")
            print(output, "LAS ENTRADAS")
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            #if steps % print_every == 0:
        model.eval()
        for batch_idx, (data, target) in enumerate(valid_loader):
#             data, target = data.cuda(), target.cuda()
#             # move to GPU
#             if use_cuda:
#                 data, target = data.cuda(), target.cuda()
                ## update the average validation loss
                # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # update running validation loss 
            valid_loss += loss.item()
            print("alexxxxxxxxx")

        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        ## TODO: save the model if validation loss has decreased
        if valid_loss < valid_loss_min:
            torch.save(model.state_dict(), save_path)
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
            valid_loss_min = valid_loss
            valid_loss = 0


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
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 10)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
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
    criterion = torch.nn.BCELoss()

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
    
    
    
    
    