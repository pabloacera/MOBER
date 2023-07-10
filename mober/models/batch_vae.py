import torch.nn as nn
import torch
from torch.distributions import Normal


class Encoder(nn.Module):
    """
    Encoder that takes the original gene expression and produces the encoding.

    Consists of 3 FC layers.
    """
    def __init__(self, n_genes, enc_dim):
        super().__init__()
        self.activation = nn.SELU()
        self.fc1 = nn.Linear(n_genes, 256)
        self.bn1 = nn.BatchNorm1d(256, momentum=0.01, eps=0.001)
        self.dp1 = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128, momentum=0.01, eps=0.001)
        self.dp2 = nn.Dropout(p=0.1)
    
        self.linear_means = nn.Linear(128, enc_dim)
        self.linear_log_vars = nn.Linear(128, enc_dim)
    
    def reparameterize(self, means, stdev):
        
        return Normal(means, stdev).rsample()
        
    def encode(self, x):
        # encode
        enc = self.fc1(x)
        enc = self.bn1(enc)
        enc = self.activation(enc)
        enc = self.dp1(enc)
        enc = self.fc2(enc)
        enc = self.bn2(enc)
        enc = self.activation(enc)
        enc = self.dp2(enc)
        
        means = self.linear_means(enc)
        #print('means', means, torch.isnan(means).nonzero())
        
        log_vars = self.linear_log_vars(enc)
        #print('log_vars', log_vars, torch.isnan(log_vars).nonzero())
        
        stdev = torch.exp(0.5 * log_vars) + 1e-4
        #print('stdev', stdev, torch.isnan(stdev).nonzero())
        z = self.reparameterize(means, stdev)
        
        return means, stdev, z
        
    def forward(self, x):
        return self.encode(x)


class Decoder(nn.Module):
    """
    A decoder model that takes the encodings and a batch (source) matrix and produces decodings.
    Made up of 3 FC layers.
    """
    def __init__(self, n_genes, enc_dim, n_batch):
        super().__init__()
        self.activation = nn.SELU()
        #self.final_activation = nn.ReLU()
        self.final_activation_mu = nn.Softplus()
        self.final_activation_theta = nn.Softplus()
        self.fcb = nn.Linear(n_batch, n_batch)
        self.bnb = nn.BatchNorm1d(n_batch, momentum=0.01, eps=0.001)
        self.fc4 = nn.Linear(enc_dim + n_batch, 128)
        self.bn4 = nn.BatchNorm1d(128, momentum=0.01, eps=0.001)
        self.fc5 = nn.Linear(128, 256)
        self.bn5 = nn.BatchNorm1d(256, momentum=0.01, eps=0.001)

        #self.out_fc = nn.Linear(256, n_genes)
        self.fc5_mu1 = nn.Linear(256, 512)
        self.bn5_mu1 = nn.BatchNorm1d(512, momentum=0.01, eps=0.001)
        self.fc5_mu2 = nn.Linear(512, 1024)
        self.bn5_mu2 = nn.BatchNorm1d(1024, momentum=0.01, eps=0.001)
        
        self.fc5_theta1 = nn.Linear(256, 512)
        self.bn5_theta1 = nn.BatchNorm1d(512, momentum=0.01, eps=0.001)
        self.fc5_theta2 = nn.Linear(512, 1024)
        self.bn5_theta2 = nn.BatchNorm1d(1024, momentum=0.01, eps=0.001)
        
        self.out_mu = nn.Linear(1024, n_genes)
        self.out_alpha = nn.Linear(1024, n_genes)


    def forward(self, z, batch):
        # batch input
        b = self.fcb(batch)
        b = self.bnb(b)
        b = self.activation(b)

        # concat with z
        n_z = torch.cat([z, b], dim=1)

        # decode layers
        dec = self.fc4(n_z)
        dec = self.bn4(dec)
        dec = self.activation(dec)
        dec = self.fc5(dec)
        dec = self.bn5(dec)
        dec = self.activation(dec)
        
        #dec = self.final_activation(self.out_fc(dec))
        mu = self.fc5_mu1(dec)
        mu = self.bn5_mu1(mu)
        mu = self.activation(mu)

        mu = self.fc5_mu2(mu)
        mu = self.bn5_mu2(mu)
        mu = self.activation(mu)
        
        theta = self.fc5_theta1(dec)
        theta = self.bn5_theta1(theta)
        theta = self.activation(theta)
        
        theta = self.fc5_theta2(theta)
        theta = self.bn5_theta2(theta)
        theta = self.activation(theta)
        
        
        mu = self.final_activation_mu(self.out_mu(mu))
        theta = self.final_activation_theta(self.out_alpha(theta))
        
        return mu, theta

class BatchVAE(nn.Module):
    """
    Batch Autoencoder.
    Encoder is composed of 3 FC layers.
    Decoder is symmetrical to encoder + Batch input.
    """
    
    def __init__(self, n_genes, enc_dim, n_batch):
        super().__init__()

        self.encoder = Encoder(n_genes, enc_dim)
        self.decoder = Decoder(n_genes, enc_dim, n_batch)

    def forward(self, x, batch):
        means, stdev, enc = self.encoder(x)
        mu, theta = self.decoder(enc, batch)
        dec =  torch.stack((mu, theta), dim=1)
        
        return dec, enc, means, stdev
    
    
    
    
    
    
    
    
    
