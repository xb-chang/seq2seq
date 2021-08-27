import torch.nn as nn

from math import exp

# def init_weights(m, w_min, w_max):
def init_weights(m):
    for name, param in m.named_parameters():
        # nn.init.uniform_(param.data, w_min, w_max)
        nn.init.uniform_(param.data, -0.08, 0.08)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

class adj_tfr_exp():
    '''
    exponentially change the teacher forcing rate, reach the 0/1 bound at 'bound_step'.
    down: direction
    True -> 1 to 0 (default)
    False -> 0 to 1
    '''
    def __init__(self, bound_step, alpha=-3, init_steps=-1, down=True) -> None:
        self.down = down
        assert bound_step > 0, 'unknown bound step: {}'.format(bound_step)
        self.steps=init_steps
        self.bound_step=bound_step
        assert alpha < 0., 'gamma rate control < 0, now at {}'.format(alpha)
        # fact: exp(-7) = 0.000911 < 10-3 (decrease faster)
        # fact: exp(-3) = 0.05
        # fact: exp(-1.4) = 0.25
        # fact: exp(-0.7) = 0.5 (decrease slower)
        self.gamma = alpha / bound_step
    
    def update(self):
        # starts from 0
        self.steps += 1
        # changed at fixed steps
        return self.get_tfr()
    
    def get_tfr(self):
        if self.steps < self.bound_step:
            prob = exp(self.gamma * self.steps)
            assert 0. < prob <= 1., prob
            if self.down:
                return prob
            else:
                return 1. - prob
        else:
            if self.down:
                return 0.
            else:
                return 1.

class adj_tfr_lin():
    '''
    linearly change the teacher forcing rate, reach the 0/1 bound at 'bound_step'.
    '''
    def __init__(self, bound_step, init_steps=-1, init_rate=1., down=True) -> None:
        assert 0 <= init_rate <=1, 'unknown teacher forcing rate: {}'.format(init_rate)
        assert bound_step > 0, 'unknown bound step: {}'.format(bound_step)
        self.steps=init_steps
        self.ini_tfr = init_rate
        if down is True:
            self.r = -1. * init_rate / bound_step
        else:
            self.r = (1. - init_rate) / bound_step

    def update(self):
        # starts from 0
        self.steps += 1
        # changed at fixed steps
        return self.get_tfr()
    
    def get_tfr(self):
        tfr = self.ini_tfr + self.r * self.steps
        if tfr < 0:
            tfr = 0.
        if tfr > 1.:
            tfr = 1.
        return tfr


class adj_tfr_lin_step():
    '''
    linearly change the teacher forcing rate at fixed step size
    '''
    def __init__(self, step_size, gamma, init_steps=-1, init_rate=1., down=True) -> None:
        assert 0 <= init_rate <=1, 'unknown teacher forcing rate: {}'.format(init_rate)
        assert 0 <= gamma <=1, 'unknown gamma: {}'.format(gamma)
        self.ini_tfr = init_rate
        self.gamma = gamma
        self.step_size = step_size
        self.steps=init_steps
        self.direct = 1.0
        if down is True:
            # teacher force rate decrease 
            self.direct = -1.0
    
    def update(self):
        # starts from 0
        self.steps += 1
        # changed at fixed steps
        return self.get_tfr()

    def get_tfr(self):
        n = self.steps // self.step_size
        tfr = self.ini_tfr + self.direct * n * self.gamma
        
        if tfr < 0:
            tfr = 0.
        if tfr > 1.:
            tfr = 1.
        
        return tfr
