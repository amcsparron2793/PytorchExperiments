#! python3
from os import system

import torch
import torch.nn as nn
import torch.optim as optim
from json import dump, dumps

# Define the model
class AnimalClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 8)
        self.fc2 = nn.Linear(8, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


# Define the training data
training_data = [
    (torch.Tensor([2, 30]), torch.Tensor([0, 1])),
    (torch.Tensor([4, 45]), torch.Tensor([1, 0])),
    (torch.Tensor([3, 35]), torch.Tensor([0, 1])),
    (torch.Tensor([5, 50]), torch.Tensor([1, 0])),
    (torch.Tensor([2.5, 32]), torch.Tensor([0, 1])),
]


def RandTrainingData(rand_species: bool, **kwargs):
    from random import randint

    try:
        if (kwargs['num_records'] % 2) == 0:
            num_records = kwargs['num_records']
            print(f"Creating {num_records} records.")
        else:
            raise ValueError("num_records must be divisible by 2")
    except UnboundLocalError:
        num_records = 500
    except KeyError:
        num_records = 500

    species_dict = [
        {
            'id': 0,
            'species': 'Cat',
            # division is to go from lbs to kg
            'weight_range': (3, 11), #(int(round((8/2.2), 0)), int(round((30/2.2), 0))),
            'height_range': (23, 25),
        },
        {
            'id': 1,
            'species': 'Dog',
            # division is to go from lbs to kg
            'weight_range': (3, 113),#(int(round((15/2.2), 0)), int(round((120/2.2), 0))),
            'height_range': (20, 76)#(40, 60)
        }
    ]

    t_data = []
    try:
        if not rand_species and not kwargs['species']:
            raise AttributeError("rand_species must be set to True if no species is given!")
        elif rand_species and kwargs['species']:
            raise AttributeError("rand_species cannot be true if species is populated!")
    except UnboundLocalError:
        if not rand_species:
            raise AttributeError("rand_species must be set to True if no species is given!")
        else:
            pass
    except KeyError:
        if not rand_species:
            raise AttributeError("rand_species must be set to True if no species is given!")
        else:
            pass

    x = 0
    while x < num_records:
        if not rand_species:
            species = kwargs['species']
            species_int = [x['id'] for x in species_dict if (species == x['species'])][0]
        else:
            species_int = species_dict[randint(0, 1)]['id']
            species = species_dict[species_int]['species']

        weight_range = tuple([x['weight_range'] for x in species_dict if species_int == x['id']][0])
        height_range = tuple([x['height_range'] for x in species_dict if species_int == x['id']][0])

        weight_int = randint(weight_range[0], weight_range[1])
        height_int = randint(height_range[0], height_range[1])

        # print(weight_int, height_int, species_int)
        t_data.append((torch.Tensor([weight_int, height_int]), torch.Tensor([species_int, abs(species_int - 1)])))
        x += 1
    return t_data


def TrainModel(model, optimizer, loss_fn, training_data):
    # Train the model
    for epoch in range(100):
        print(f"Training epoch: {epoch}")
        system("cls")

        for inputs, labels in training_data:
            # Clear the gradients
            optimizer.zero_grad()

            # Make a forward pass through the model
            outputs = model(inputs)
            # print(outputs)
            # Compute the loss
            loss = loss_fn(outputs, labels)

            # Backpropagate the error
            loss.backward()

            # Update the model's parameters
            optimizer.step()


if __name__ == "__main__":
    """DogData = RandTrainingData(rand_species=False, species="Dog", num_records=10)
    CatData = RandTrainingData(rand_species=False, species="Cat", num_records=10)
    #training_data = CatData + DogData"""

    print(f"{len(training_data)} training records detected.")
    # Create the model and optimizer
    model = AnimalClassifier()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    TrainModel(model, optimizer, loss_fn, training_data)

    # Test the model
    #inputs = torch.Tensor([3.5, 37])
    inputs = torch.Tensor([55, 25])
    prediction = model(inputs)
    print(f"cat likelihood: {round(float(prediction[0]), 2) * 100}%"
          f"\ndog likelihood: {round(float(prediction[1]), 2) * 100}%")
    print(prediction)

