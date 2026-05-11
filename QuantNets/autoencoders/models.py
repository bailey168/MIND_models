# models.py
import torch
import torch.nn as nn

class AgeGuidedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dims, age_predictor_hidden_dims, age_predictor_dropout):
        super().__init__()
        # encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ELU()])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim), nn.ELU()])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

        # age predictor
        age_predictor_layers = []
        prev_dim = latent_dim
        for h_dim in age_predictor_hidden_dims:
            age_predictor_layers.extend([
                nn.Linear(prev_dim, h_dim), nn.BatchNorm1d(h_dim),
                nn.ELU(), nn.Dropout(age_predictor_dropout)
            ])
            prev_dim = h_dim
        age_predictor_layers.append(nn.Linear(prev_dim, 1))
        self.age_predictor = nn.Sequential(*age_predictor_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        age_pred = self.age_predictor(z).squeeze(-1)
        return x_hat, z, age_pred
    

class AgeGuidedLoss(nn.Module):
    def __init__(self, recon_weight, age_weight):
        super().__init__()
        self.recon_weight = recon_weight
        self.age_weight = age_weight
        self.mse_loss = nn.MSELoss()

    def forward(self, x, x_hat, age_target, age_pred):
        recon_loss = self.mse_loss(x_hat, x)
        age_loss = self.mse_loss(age_pred, age_target)
        return {
            'total_loss': self.recon_weight * recon_loss + self.age_weight * age_loss,
            'recon_loss': recon_loss, 
            'age_loss': age_loss
        }