import torch 
from dataclasses import dataclass, field
@dataclass
class freenerfT:
    max_num_iterations: int =30000

def get_freq_mask(pos_enc_length, current_iter, total_reg_iter):
    if current_iter < total_reg_iter:
        freq_mask=torch.zeros(pos_enc_length)
        ptr = pos_enc_length / 3 * current_iter / total_reg_iter + 1
        ptr = ptr if ptr < pos_enc_length / 3 else pos_enc_length / 3
        int_ptr = int(ptr)
        freq_mask[: int_ptr * 3] = 1.0  # assign the integer part
        freq_mask[int_ptr * 3 : int_ptr * 3 + 3] = (ptr - int_ptr)  # assign the fractional part
        return torch.clip(freq_mask,1e-8, 1-1e-8) # for math stabiltiy
    else:
        return torch.ones(pos_enc_length)
    
