import torch.optim as optim
from torch.utils.data import DataLoader
from copy import deepcopy

class Client:
    def __init__(self, client_id, model, dataset, device, batch_size, weight, correction):
        self.client_id = client_id
        self.model = deepcopy(model).to(device)
        self.memory = deepcopy(model.state_dict())
        # Saved version of the server model
        self.x_s_i = deepcopy(model.state_dict())
        self.data_loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        self.device = device
        self.weight = weight
        self.correction = correction
        
    def client_update(self, alpha, criterion, num_local_steps=1):
        
        if self.correction:
            return self.area_update(alpha, criterion, num_local_steps)
        else:
            return self.uncorrected_update(alpha, criterion, num_local_steps)

    def area_update(self, alpha, criterion, num_local_steps):
        # Load the saved server model
        self.model.load_state_dict(deepcopy(self.x_s_i))
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

        # Calculate x_i
        x_i = deepcopy(self.model.state_dict())

        # Calculate the difference x_i - self.memory
        update = {key: x_i[key] - self.memory[key] for key in x_i.keys()}

        # Update memory with x_i
        self.memory = x_i

        return update

    def uncorrected_update(self, alpha, criterion, num_local_steps=1):
        # Load the saved server model
        self.model.load_state_dict(deepcopy(self.x_s_i))
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

        # Calculate x_i
        x_i = deepcopy(self.model.state_dict())

        # Calculate the difference x_i - x_s_i
        update = {key: x_i[key] - self.x_s_i[key] for key in x_i.keys()}

        return update

    def receive_global_model(self, global_model):
        # Update the saved server model with the latest global model
        self.x_s_i = deepcopy(global_model)
