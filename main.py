from ppo import Model
from env import Env
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import copy
import numpy as np

torch.set_default_tensor_type('torch.cuda.FloatTensor')

def choose_action(net, state):
    tensor_state = Variable(torch.from_numpy(state).float())
    action, value = net(tensor_state)
    action = F.softmax(action, dim=1)
    action = action.data.cpu().numpy()[0]
    print("policy:", action)
    length_action = len(action)
    value = value.data.cpu().numpy()[0]
    action = np.random.choice(length_action, p=action)
    action = np.eye(length_action)[action]
    return action, value

def assign_parameter(target_network, main_network):
    target_network.load_state_dict(main_network.state_dict())

def get_next_vpreds(v_preds):
    return v_preds[1:] + [np.array([0], dtype=float)]

def get_gaes(rewards, v_preds, v_preds_next):
    deltas = [r_t + 0.99 * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
    # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
    gaes = copy.deepcopy(deltas)
    for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
        gaes[t] = gaes[t] + 0.99 * gaes[t + 1]
    return gaes

def train(Old_Policy, Policy, observations, actions, rewards, v_preds_next, gaes):
    tensor_observations = Variable(torch.from_numpy(observations)).float().cuda()
    tensor_actions = Variable(torch.from_numpy(actions)).float().cuda()
    tensor_rewards = Variable(torch.from_numpy(rewards)).float().cuda()
    tensor_v_preds_next = Variable(torch.from_numpy(v_preds_next)).float().cuda()
    tensor_gaes = Variable(torch.from_numpy(gaes)).float().cuda()
    
    pi, value = Policy(tensor_observations)
    value = value.squeeze(1)
    pi_old, _ = Old_Policy(tensor_observations)
    
    prob = F.softmax(pi, dim=1)
    log_prob = F.log_softmax(pi, dim=1)
    action_prob = torch.sum(torch.mul(prob, tensor_actions), dim=1)
    
    prob_old = F.softmax(pi_old, dim=1)
    action_prob_old = torch.sum(torch.mul(prob_old, tensor_actions), dim=1)
    
    ratio = action_prob / (action_prob_old + 1e-10)
    advantage = (tensor_gaes - tensor_gaes.mean()) / (tensor_gaes.std() + 1e-5)
    
    surr1 = ratio * advantage
    surr2 = torch.clamp(ratio, min=1. - 0.2, max=1. + 0.2) * advantage
    
    policy_loss = -torch.min(surr1, surr2).mean()
    value_diff = tensor_v_preds_next * 0.99 + tensor_rewards - value
    value_loss = torch.mul(value_diff, value_diff).mean()
    entropy_loss = (prob * log_prob).sum(1).mean()
    
    total_loss = policy_loss + value_loss + entropy_loss * 0.001
    print("policy loss:", policy_loss.data)
    print("value loss:", value_loss.data)
    print("entropy loss:", entropy_loss.data)
    optimizer = torch.optim.Adam(Policy.parameters(), lr=3e-4)
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

writer = SummaryWriter()
Policy = Model().cuda()
Old_Policy = Model().cuda()
env = Env()

for episodes in range(100000):
    done = False
    state = env.reset()
    observations, actions, rewards, v_preds = [], [], [], []
    global_step = 0
    while not done:
        global_step += 1
        action, value = choose_action(Policy, state)
        next_state, reward, done = env.step(np.argmax(action))

        observations.append(state)
        v_preds.append(value)
        actions.append(action)
        rewards.append(reward)
        state = next_state

    v_preds_next = get_next_vpreds(v_preds)
    gaes = get_gaes(rewards, v_preds, v_preds_next)

    observations = np.array(observations)
    actions = np.array(actions)
    rewards = np.array(rewards)
    v_preds_next = np.array(v_preds_next)
    gaes = np.array(gaes)

    train(observations=observations,
        actions=actions,
        rewards=rewards,
        v_preds_next=v_preds_next,
        gaes=gaes,
        Old_Policy=Old_Policy,
        Policy=Policy)
    assign_parameter(Old_Policy, Policy)
    writer.add_scalar('data/reward', np.sum(rewards), episodes)
    print(episodes, global_step)
    torch.save(Policy.state_dict(), 'models/model.pth')

env.close()