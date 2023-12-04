import torchvision;
import torch.utils.data.dataloader
import torch.nn;
from tqdm import tqdm;
import os;
import numpy as np;
import copy;

# Configurations
TRAINSET_DIR = './train';
TESTSET_DIR = './test';
BATCH_SIZE = 30;
MODEL_SAVES_DIR = './saves';
EPOCHS = 80;

# Load the dataset
trainset = torchvision.datasets.FashionMNIST(TRAINSET_DIR, train=True, 
                                                transform=torchvision.transforms.ToTensor(), 
                                                download=True);
testset = torchvision.datasets.FashionMNIST(TESTSET_DIR, train=False, 
                                                transform=torchvision.transforms.ToTensor(), 
                                                download=True);
# dataloaders leverage multiprocessing to fetch data efficiently
trainloader = torch.utils.data.dataloader.DataLoader(testset, BATCH_SIZE);
testloader = torch.utils.data.dataloader.DataLoader(testset, BATCH_SIZE);

# neural network architecture we will use to predict labels
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__();

        # Hidden Layer 1: 32 Filters
        self.conv1 = torch.nn.Conv2d(
            in_channels = 1, out_channels = 32, kernel_size = (3, 3));
        self.reLU1 = torch.nn.ReLU();
        self.maxpool1 = torch.nn.MaxPool2d((2, 2));

        # Hidden Layer 2: 64 Filters
        self.conv2 = torch.nn.Conv2d(
            in_channels = 32, out_channels = 64, kernel_size = (3, 3));
        self.reLU2 = torch.nn.ReLU();
        self.maxpool2 = torch.nn.MaxPool2d((2, 2));

        # Hidden Layer 3: 64 Filters
        self.conv3 = torch.nn.Conv2d(
            in_channels = 64, out_channels = 64, kernel_size = (3, 3));
    
        # Flatten to 1D vector
        self.flatten1 = torch.nn.Flatten(1);
    
        # Linear layers
        self.linear1 = torch.nn.Linear(576, 250);
        self.linear1Activation = torch.nn.ReLU();
        self.linear2 = torch.nn.Linear(250, 125);
        self.linear2Activation = torch.nn.ReLU();
        self.linear3 = torch.nn.Linear(125, 60);
        self.linear3Activation = torch.nn.ReLU();
        self.linear4 = torch.nn.Linear(60, 10);
        self.linear4Activation = torch.nn.ReLU();
        self.linear4Clip = torch.nn.Softmax();

    def forward(self, X):
        X = self.conv1(X);
        X = self.reLU1(X);
        X = self.maxpool1(X);
        X = self.conv2(X);
        X = self.reLU2(X);
        X = self.maxpool2(X);
        X = self.conv3(X);
        X = self.flatten1(X);
        X = self.linear1(X);
        X = self.linear1Activation(X);
        X = self.linear2(X);
        X = self.linear2Activation(X);
        X = self.linear3(X);
        X = self.linear3Activation(X);
        X = self.linear4(X);
        X = self.linear4Activation(X);
        X = self.linear4Clip(X);
        return X;


if os.listdir(MODEL_SAVES_DIR) == []:

    net = Net();
    net.train();

    loss_function = torch.nn.CrossEntropyLoss();
    optimizer = torch.optim.Adam(net.parameters());

    for epoch in range(EPOCHS):
        for i, data in tqdm(enumerate(trainloader)):
            optimizer.zero_grad();

            # get predictions
            inputs, labels = data;
            logits = net(inputs);

            # compute loss, compute gradients, update gradients accordingly
            loss = loss_function(logits, labels);
            loss.backward();
            optimizer.step();

    # save/load model mechanism to avoid re-running while debugging later parts
    torch.save(net, f'{MODEL_SAVES_DIR}/last_save.pt');
else:
    net = torch.load(f'{MODEL_SAVES_DIR}/last_save.pt');

# Evaluating the model's performance on test data
net.train(mode = False);
correct = 0;
incorrect = 0;
for i, data in tqdm(enumerate(testloader)):
    inputs, labels = data;
    logits = net(inputs);
    predictions = torch.argmax(logits, 1);
    comparisons = torch.eq(predictions, labels);
    num_correct = comparisons.long().count_nonzero();
    correct += num_correct;
    incorrect += BATCH_SIZE - num_correct;

print(f'Accuracy: {correct / (correct + incorrect)}');

###########################################
# Calculating sensitivity
###########################################

# Considering only the first layer for now

# Compute sensitivities
kernels = net.conv1.weight;
sensitivities = torch.zeros_like(kernels);  # sensitivity scores
print('Computing sensitivites...')
for k in range(kernels.shape[0]):
    kernel = torch.stack([kernels[k]]);
    # for one batch of the data,
    for i, data in enumerate(trainloader):
        images = data[0];
        old_featmaps = torch.nn.functional.conv2d(images, kernel);  
        for i, j in np.ndindex(kernel.shape[2:]):
            new = kernel.detach().clone();
            new[0, 0, i, j] = 0;
            new_featmaps = torch.nn.functional.conv2d(images, new);
            sensitivities[k, 0, i, j] = sensitivities[k, 0, i, j] \
                + torch.linalg.matrix_norm(old_featmaps - new_featmaps, dim=(2, 3)).sum().item();

# Sort sensitivites
sensitivities_as_list = []
for k in range(kernels.shape[0]):
    for i, j in np.ndindex(kernel.shape[2:]):
        sensitivities_as_list.append((k, i, j, sensitivities[k, 0, i, j]));
sorted_sensitivities = sorted(sensitivities_as_list, key=lambda item : item[3]);

# Prune bottom 10% of sensitivities
pruning_mass = torch.numel(kernels) // 10
pruned = kernels.detach().clone();
for idx in range(pruning_mass):
    weight = sorted_sensitivities[idx];
    pruned[weight[0], 0, weight[1], weight[2]] = 0;
pruned_net = copy.deepcopy(net);
pruned_net.conv1.weight = torch.nn.Parameter(pruned, requires_grad=False);

# Evaluating pruned model's performance on test data
pruned_net.train(mode = False);
new_correct = 0;
new_incorrect = 0;
print('Evaluating new performance...')
for i, data in tqdm(enumerate(testloader)):
    inputs, labels = data;
    logits = pruned_net(inputs);
    predictions = torch.argmax(logits, 1);
    comparisons = torch.eq(predictions, labels);
    num_correct = comparisons.long().count_nonzero();
    new_correct += num_correct;
    new_incorrect += BATCH_SIZE - num_correct;

print(f'Accuracy: {new_correct / (new_correct + new_incorrect)}');