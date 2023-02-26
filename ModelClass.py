import torch
import torch.nn as nn
import io
import torchvision.transforms as transforms
from PIL import Image

class MultilayerRNN_MNIST(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, relu=True):
        super(MultilayerRNN_MNIST, self).__init__()
        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        
        # Create RNN
        if relu:
            self.rnn = nn.RNN(input_size, hidden_size, layer_size, batch_first=True, nonlinearity='relu')
        else:
            self.rnn = nn.RNN(input_size, hidden_size, layer_size, batch_first=True, nonlinearity='tanh')
            
        # Create FNN
        self.fnn = nn.Linear(hidden_size, output_size)
        
    def forward(self, images, prints=False):
        if prints: print('images shape:', images.shape)
        
        # Instantiate hidden_state at timestamp 0
        hidden_state = torch.zeros(self.layer_size, images.size(0), self.hidden_size)
        hidden_state = hidden_state.requires_grad_()
        if prints: print('Hidden State shape:', hidden_state.shape)
        
        # Compute RNN
        # .detach() is required to prevent vanishing gradient problem
        output, last_hidden_state = self.rnn(images, hidden_state.detach())
        if prints: print('RNN Output shape:', output.shape, '\n' +
                         'RNN last_hidden_state shape', last_hidden_state.shape)
        
        # Compute FNN
        # We get rid of the second size
        output = self.fnn(output[:, -1, :])
        if prints: print('FNN Output shape:', output.shape)
        
        return output


def transform_image(path):
    f = open(path, 'rb')
    image_bytes = f.read()
    my_transform = transforms.Compose([
        transforms.Resize([28,28]),
        transforms.Grayscale(1),
        transforms.ToTensor(),
    ])

    image = Image.open(io.BytesIO(image_bytes))
    f.close()
    tensors = my_transform(image).unsqueeze(0)
    tensor = next(iter(tensors))

    return tensor


