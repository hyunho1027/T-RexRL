import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import copy

def ortho_weights(shape, scale=1.0):
    shape = tuple(shape)
    
    if len(shape) == 2:
        flat_shape = shape[1], shape[0]
    elif len(shape) == 4:
        flat_shape = (np.prod(shape[1:]), shape[0])
    else:
        raise NotImplementedError
    
    a = np.random.normal(0., 1., flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    q = q.transpose().copy().reshape(shape)

    if len(shape) == 2:
        return torch.from_numpy((scale * q).astype(np.float32))
    if len(shape) == 4:
        return torch.from_numpy((scale * q[:, :shape[1], :shape[2]]).astype(np.float32))

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.actor_fc1 = nn.Linear(2*64*128, 128)
        self.actor_fc2 = nn.Linear(128, 64)
        self.actor_fc3 = nn.Linear(64, 2)
        
        self.critic_fc1 = nn.Linear(2*64*128, 128)
        self.critic_fc2 = nn.Linear(128, 64)
        self.critic_fc3 = nn.Linear(64, 1)
        
        self.actor_fc3.weight.data = ortho_weights(self.actor_fc3.weight.size(), scale=1.0)
        self.critic_fc3.weight.data = ortho_weights(self.critic_fc3.weight.size(), scale=1.0)
    
    
    def forward(self, x):
        x = x.view(-1, 2*64*128).cuda()
        actor = F.relu(self.actor_fc1(x))
        actor = F.relu(self.actor_fc2(actor))
        actor = self.actor_fc3(actor)
        
        critic = F.relu(self.critic_fc1(x))
        critic = F.relu(self.critic_fc2(critic))
        critic = self.critic_fc3(critic)
        
        return actor, critic
