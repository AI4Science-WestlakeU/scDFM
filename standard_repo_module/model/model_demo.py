import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
class Net_demo(nn.Module):
    '''
        simulation model
            input: u_0 (B,1,T,d) 
            output: u_[1:T] (B,1,T,d)
    '''
    def __init__(self, h=256,input_channel=80,channels=[128,256,256],output_channel=80):
        super(Net_demo, self).__init__()
        if h%4==0:
        	self.h = h // 4
        else:
        	self.h = 30
        self.channels= channels
        self.down = nn.Sequential(
            nn.Conv1d(input_channel, channels[0], 5, padding=2),
            nn.ELU(),
            nn.Conv1d(channels[0],channels[1], 5, stride=2, padding=2), 
            nn.ELU(),
            nn.Conv1d(channels[1], channels[2], 5, stride=2, padding=2), 
            nn.ELU(),
        )
        self.enc=nn.Linear(256*32,256*128)####

        self.up = nn.Sequential(
            nn.Conv1d(channels[2], channels[1], 5, padding=2),
            nn.ELU(),
            nn.Conv1d(channels[1],channels[0], 5, padding=2),
            nn.ELU(),
            nn.Conv1d(channels[0], output_channel, 5, padding=2),
            nn.ELU(),
        )
        self.dec=nn.Linear(1,2)####
        
    def forward(self, u):
        '''
        u: (B, 1,T, d)
        u_next: (B,1, T, d)
        '''
        bs,_,c,d=u.shape
        u = u.squeeze()
        u_latent = self.down(u) # [B,256,32]

        u_latent= rearrange(u_latent,"b c d -> b (c d)")
        u_latent = self.enc(u_latent)
        u_latent= rearrange(u_latent,"b (c d) -> b c d",c=self.channels[-1]) # [B,256,128]

        u_next = self.up(u_latent)
        u_next = u_next.view(bs,1,c,d)

        return u_next