import torch
import numpy as np
import os
from typing import Any, Dict, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   

test_recurrent_hidden_states = torch.zeros(1, *(0, 512))

prev_actions = torch.zeros(1,*np.array([[0,0],[0,0]]).shape)
print(f"prev_actions shape: {prev_actions.shape},test_recurrent_hidden_states shape: {test_recurrent_hidden_states.shape}")
rgb_frames: List[List[np.ndarray]] = [[] for _ in range(2)]
print(f"rgb_frames: {rgb_frames}")
# not_done_masks = torch.zeros(1,1,device=device,dtype=torch.bool)]