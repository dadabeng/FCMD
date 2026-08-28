import torch
import torch.nn as nn


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class Film_Layer(nn.Module):
    def __init__(self, latent_dim, time_embed_dim, dropout=0):
        super().__init__()
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, 2 * latent_dim),
        )
        self.norm = nn.LayerNorm(latent_dim)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Linear(latent_dim, latent_dim)),
        )

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, D
        """
        # B, 1, 2D
        emb_out = self.emb_layers(emb).unsqueeze(1)
        # scale: B, 1, D / shift: B, 1, D
        scale, shift = torch.chunk(emb_out, 2, dim=2)
        h = self.norm(h) * (1 + scale) + shift
        h = self.out_layers(h)
        return h        

class Local_Text_Fuse(nn.Module):

    def __init__(self, latent_dim, time_embed_dim, dropout=0):
        super().__init__()
        self.layer_0 =  Film_Layer(latent_dim, time_embed_dim)
        self.layer_1 =  Film_Layer(latent_dim, time_embed_dim)
        self.layer_2 =  Film_Layer(latent_dim, time_embed_dim)
        self.layer_3 =  Film_Layer(latent_dim, time_embed_dim)
        self.layer_4 =  Film_Layer(latent_dim, time_embed_dim)
        self.layer_5 =  Film_Layer(latent_dim, time_embed_dim)

    def forward(self, h, emb):
        """
        h: B, T, D
        emb: B, N, D
        """
        h = h+self.layer_0(h,emb[:,0])
        h = h+self.layer_1(h,emb[:,1])
        h = h+self.layer_2(h,emb[:,2])
        h = h+self.layer_3(h,emb[:,3])
        h = h+self.layer_4(h,emb[:,4])
        h = h+self.layer_5(h,emb[:,5])
        
        return h
    
