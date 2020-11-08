# torch imports
import torch.nn.functional as F
import torch.nn as nn



num_classes = 133
# define the CNN architecture
class Net(nn.Module):
    def __init__(self, input_features, hidden_dim, output_dim):
        super(Net, self).__init__()
        ## Define layers of a CNN
        self.conv1 = nn.Conv2d(3, 32, 3, stride = 2, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 2, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding = 1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully-connected layers
        self.fc1 = nn.Linear(7 * 7 * 128, 500)
        self.fc2 = nn.Linear(500, num_classes)
        
        #Dropout
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        ## Define forward behavior
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        
        x = x.view(-1, 7 * 7 * 128)
        
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        
        return x
    
# class Net(nn.Module):
#     def __init__(self, input_features, hidden_dim, output_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(7 * 7 * 128, 500)
#         self.fc2 = nn.Linear(500, 256)
#         self.fc3 = nn.Linear(256, 150)
#         self.fc4 = nn.Linear(150, 133)
        
#     def forward(self, x):
#         # make sure input tensor is flattened
#         x = x.view(x.shape[0], -1)
        
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.log_softmax(self.fc4(x), dim=1)
        
#         return x