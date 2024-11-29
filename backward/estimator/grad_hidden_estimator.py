import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class Grad_hidden_checker():
    def __init__(self, args, optimizer, reduction='sum'):
        # TODO: sum is better than the usual one
        self._optim, self._reduction = optimizer, reduction
        self.args = args
        

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()
    
    def save_grad(self, model, main_loss, smooth_losses):        
        main_grad = torch.autograd.grad(main_loss, model.hidden, retain_graph=True)[0] # .detach() retain_graph=True
        smooth_grads = []
        
        for smooth_loss in smooth_losses:
            # import ipdb; ipdb.set_trace()   
            self._optim.zero_grad()
            smooth_grads.append(torch.autograd.grad(smooth_loss, model.hidden, retain_graph=True)[0]) # .detach() retain_graph=True
        self._optim.zero_grad()

        return main_grad, smooth_grads
        
    def _clean_grad(self, model):
        for hidden in model.encoder.hidden_list:
            hidden.grad = None

    def _pack_grad(self, main_loss, smooth_losses):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''
        # self.args.logger.info(f"we have {len(smooth_losses)} smoothness losses")
        
        main_grad = torch.autograd.grad(main_loss, model.hidden_before, retain_graph=True)[0] # .detach() retain_graph=True
        smooth_grads = []
        
        for smooth_loss in smooth_losses:
            self._optim.zero_grad()
            # smooth_loss.backward() 
            smooth_grads.append(torch.autograd.grad(smooth_loss, model.hidden_before, retain_graph=True)[0]) # .detach() retain_graph=True
        self._optim.zero_grad()

        return main_grads, smooth_grads
        
    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self, model):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        # self.args.logger.debug(f"{self._optim.param_groups}")
        # import ipdb; ipdb.set_trace()
        for hidden in model.encoder.hidden_list:            
            #TODO: this place may need modification
            # print(hidden.grad.shape)
            if hidden.grad is None:
                shape.append(hidden.shape)
                hidden.grad.append(torch.zeros_like(hidden).to(hidden.device))
                has_grad.append(torch.zeros_like(hidden).to(hidden.device))
                continue
            shape.append(hidden.grad.shape)
            grad.append(hidden.grad.clone())
            has_grad.append(torch.ones_like(hidden).to(hidden.device))
        return grad, shape, has_grad

        '''
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                # NOTICE!!!!!, we do not compare on the bias matric
                if len(p.shape) == 1:
                    continue
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad
        '''

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives, loss function
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        # TODO: we need to specfic what is the main task and which
        shared = torch.stack(has_grads).prod(0).bool()
        # return the product of all elements in the input tensor
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    # TODO: different method
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return


