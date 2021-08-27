import torch.nn as nn

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
