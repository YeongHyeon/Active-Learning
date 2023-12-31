import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import source.utils as utils

import torchsummary
from torch.utils.tensorboard import SummaryWriter

class Agent(object):

    def __init__(self, **kwargs):

        print("\nInitializing Neural Network...")
        self.nn = kwargs['nn']
        self.dim_h = kwargs['dim_h']
        self.dim_w = kwargs['dim_w']
        self.dim_c = kwargs['dim_c']
        self.num_class = kwargs['num_class']
        self.k_size = kwargs['k_size']
        self.filters = kwargs['filters']
        tmp_filters = self.filters.split(',')
        for idx, _ in enumerate(tmp_filters):
            tmp_filters[idx] = int(tmp_filters[idx])
        self.filters = tmp_filters

        self.learning_rate = kwargs['learning_rate']
        self.path_ckpt = kwargs['path_ckpt']

        self.ngpu = kwargs['ngpu']
        self.device = kwargs['device']

        self.__model = self.nn.Neuralnet(dim_h=self.dim_h, dim_w=self.dim_w, dim_c=self.dim_c, num_class=self.num_class, \
            k_size=self.k_size, filters=self.filters, \
            learning_rate=self.learning_rate, path_ckpt=self.path_ckpt, \
            ngpu=self.ngpu, device=self.device).to(self.device)

        self.__init_propagation(path=self.path_ckpt)
        self.softmax = nn.Softmax(dim=1)

    def __init_propagation(self, path):

        utils.make_dir(self.path_ckpt, refresh=False)
        self.save_params()

        out = torchsummary.summary(self.__model, (self.dim_c, self.dim_h, self.dim_w))

        self.optimizer = optim.Adam(self.__model.parameters(), lr=self.learning_rate)
        self.writer = SummaryWriter(log_dir=self.path_ckpt)

    def step(self, minibatch, iteration=0, training=False):

        x, y = minibatch['x'], minibatch['y']
        x, y = torch.tensor(utils.nhwc2nchw(x)), torch.tensor(y)
        x = x.to(self.device)
        y = y.to(self.device)

        if(training):
            self.optimizer.zero_grad()

        step_dict = self.__model(x)
        y_hat = step_dict['y_hat']
        l_hat = step_dict['l_hat']
        losses = {}
        losses.update(self.__model.loss_target_pred(y, y_hat))
        if(training): 
            losses.update(self.__model.loss_loss_pred(losses['opt_y_b'], l_hat))
            losses['opt'] = losses['opt_y'] + losses['opt_l']

        score = self.softmax(y_hat)
        preds = torch.argmax(score, 1)
        truth = torch.argmax(y, 1)
        correct_pred = torch.eq(preds, truth)
        accuracy = torch.mean(correct_pred.to(torch.float32))

        if(training):
            losses['opt'].backward()
            self.optimizer.step()

        if(training):
            self.writer.add_scalar("%s/opt" %(self.__model.who_am_i), scalar_value=losses['opt'], global_step=iteration)
            self.writer.add_scalar("%s/lr" %(self.__model.who_am_i), scalar_value=self.optimizer.param_groups[0]['lr'], global_step=iteration)
            self.writer.add_scalar("%s/acc" %(self.__model.who_am_i), scalar_value=accuracy, global_step=iteration)

        for key in list(losses.keys()):
            losses[key] = utils.detach(losses[key])

        return {'y_hat':utils.detach(y_hat), 'l_hat':utils.detach(l_hat), 'score':utils.detach(score), 'losses':losses, 'accuracy':utils.detach(accuracy)}

    def save_params(self, model='base'):

        torch.save(self.__model.state_dict(), os.path.join(self.path_ckpt, '%s.pth' %(model)))

    def load_params(self, model):

        self.__model.load_state_dict(torch.load(os.path.join(self.path_ckpt, '%s' %(model))), strict=False)
        self.__model.eval()
