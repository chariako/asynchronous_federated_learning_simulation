import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy
import torch

class Client:
    def __init__(self, client_id, model, dataset, device, batch_size, weight, correction):
        self.client_id = client_id
        self.model = deepcopy(model).to('cpu')
        if correction:
            self.memory = deepcopy(self.model.state_dict())
        self.data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.device = device
        self.weight = weight
        self.correction = correction
        
    def client_update(self, alpha, criterion, num_local_steps=1):
        
        if self.correction:
            return self.area_update(alpha, criterion, num_local_steps)
        else:
            return self.uncorrected_update(alpha, criterion, num_local_steps)

    def area_update(self, alpha, criterion, num_local_steps):
        # Move model to device for training
        self.model = self.model.to(self.device)
        self.model.train()

        # Perform local update
        optimizer = optim.SGD(self.model.parameters(), lr=alpha * self.weight)
        for _ in range(num_local_steps):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break  

        temp = deepcopy(self.model.state_dict())
        with torch.no_grad():
            for key in self.model.state_dict().keys():
                temp[key] = self.model.state_dict()[key] - self.memory[key].to(self.device)

        self.memory = deepcopy(self.model.state_dict())
        
        return temp

    def uncorrected_update(self, alpha, criterion, num_local_steps=1):

        # Move model to device for training
        self.model = self.model.to(self.device)
        temp = deepcopy(self.model.state_dict())
        self.model.train()

        # Perform local update
        optimizer = optim.SGD(self.model.parameters(), lr=alpha * self.weight)
        for _ in range(num_local_steps):
            for inputs, labels in self.data_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                break 

        with torch.no_grad():
            for key in self.model.state_dict().keys():
                temp[key] = self.model.state_dict()[key] - temp[key]

        return temp

    def receive_global_model(self, global_model):
        # Update the saved server model with the latest global model
        self.model.load_state_dict(global_model)
        self.model = self.model.to('cpu')
