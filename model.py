# -*- coding: utf-8 -*-
import torch
import numpy as np
class RNN(torch.nn.Module):
    def __init__(self, options, place_cells):
        super(RNN, self).__init__()
        self.Ng = options.Ng
        self.Np = options.Np
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Input weights
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.RNN = torch.nn.RNN(input_size=2,
                                hidden_size=self.Ng,
                                nonlinearity=options.activation,
                                bias=False)
        # Linear read-out weights
        self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Encoder initialization - Kaiming/He initialization for linear layer
        torch.nn.init.kaiming_normal_(self.encoder.weight, nonlinearity='linear')
        
        # RNN initialization
        for name, param in self.RNN.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights
                torch.nn.init.kaiming_normal_(param, nonlinearity=self.RNN.nonlinearity)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights - use orthogonal initialization
                torch.nn.init.orthogonal_(param)
        
        # Decoder initialization
        torch.nn.init.kaiming_normal_(self.decoder.weight, nonlinearity='linear')

    def g(self, inputs):
        '''
        Compute grid cell activations.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            g: Batch of grid cell activations with shape [batch_size, sequence_length, Ng].
        '''
        v, p0 = inputs
        init_state = self.encoder(p0)[None]
        g,_ = self.RNN(v, init_state)
        return g
    

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].

        Returns: 
            place_preds: Predicted place cell activations with shape 
                [batch_size, sequence_length, Np].
        '''
        place_preds = self.decoder(self.g(inputs))
        
        return place_preds


    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Batch of 2d velocity inputs with shape [batch_size, sequence_length, 2].
            pc_outputs: Ground truth place cell activations with shape 
                [batch_size, sequence_length, Np].
            pos: Ground truth 2d position with shape [batch_size, sequence_length, 2].

        Returns:
            loss: Avg. loss for this training batch.
            err: Avg. decoded position error in cm.
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(self.predict(inputs))
        loss = -(y*torch.log(yhat)).sum(-1).mean()

        # Weight regularization 
        loss += self.weight_decay * (self.RNN.weight_hh_l0**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err



from nanogpt import LayerNorm, Block, GPTConfig
import math
import torch.nn as nn

class Transformer(torch.nn.Module):
    def __init__(self, options, place_cells):
        super().__init__()
        self.Ng = options.Ng  # Number of grid cells (embedding dimension)
        self.Np = options.Np  # Number of place cells
        self.sequence_length = options.sequence_length
        self.weight_decay = options.weight_decay
        self.place_cells = place_cells

        # Create config for transformer blocks
        self.config = GPTConfig(
            block_size=self.sequence_length,
            vocab_size=None,  # Not using token embeddings
            n_layer=1,  # Can adjust number of layers
            n_head=1,  # Can adjust number of heads
            n_embd=self.Ng,  # Using Ng as embedding dimension
            dropout=0.0,
            bias=False
        )

        # Input projections
        self.encoder = torch.nn.Linear(self.Np, self.Ng, bias=False)
        self.velocity_encoder = torch.nn.Linear(2, self.Ng, bias=False)
        
        # Transformer layers
        self.drop = torch.nn.Dropout(self.config.dropout)
        self.blocks = torch.nn.ModuleList([Block(self.config) for _ in range(self.config.n_layer)])
        self.ln_f = LayerNorm(self.config.n_embd, bias=self.config.bias)
        
        # Output projection
        # self.decoder = torch.nn.Linear(self.Ng, self.Np, bias=False)

        self.decoder1 = torch.nn.Linear(self.Ng, self.Ng, bias=False)
        self.decoder2 = torch.nn.Linear(self.Ng, self.Np, bias=False)

        self.softmax = torch.nn.Softmax(dim=-1)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def g(self, inputs):
        '''
        Compute grid cell activations using transformer.
        Args:
            inputs: Tuple of (velocity, initial_position)
                velocity: Shape [sequence_length, batch_size, 2]
                initial_position: Shape [batch_size, Np]
        Returns:
            g: Batch of grid cell activations [sequence_length, batch_size, Ng]
        '''
        v, p0 = inputs
        batch_size = v.shape[1]
        
        # Encode initial position and reshape
        place_encoding = self.encoder(p0)  # [batch_size, Ng]
        place_encoding = place_encoding.unsqueeze(1)  # [batch_size, 1, Ng]
        
        # Encode velocity sequence
        v = v.permute(1, 0, 2)  # [batch_size, sequence_length, 2]
        vel_encoding = self.velocity_encoder(v)  # [batch_size, seq_len, Ng]
        
        # Combine position and velocity info - expand pos_encoding to match vel_encoding
        x = vel_encoding + place_encoding  # [batch_size, seq_len, Ng]
        # x = torch.relu(x)
        
        # Apply transformer blocks
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        # x = self.ln_f(x)

        # x = self.decoder1(x)
        x = torch.relu(x)
        
        # Permute back to [sequence_length, batch_size, Ng]
        x = x.permute(1, 0, 2)
        return x

    def predict(self, inputs):
        '''
        Predict place cell code.
        Args:
            inputs: Same as g() method.
        Returns: 
            place_preds: Predicted place cell activations [sequence_length, batch_size, Np]
        '''
        g_output = self.g(inputs)
        # Apply decoder while maintaining [sequence_length, batch_size, Np] shape
        return self.decoder2(g_output)
        # x = self.decoder1(g_output)
        # x = self.decoder2(x)
        return x

    def compute_loss(self, inputs, pc_outputs, pos):
        '''
        Compute avg. loss and decoding error.
        Args:
            inputs: Same as g() method
            pc_outputs: Ground truth place cell activations [sequence_length, batch_size, Np]
            pos: Ground truth 2d position [sequence_length, batch_size, 2]
        Returns:
            loss: Avg. loss for this training batch
            err: Avg. decoded position error in cm
        '''
        y = pc_outputs
        preds = self.predict(inputs)
        yhat = self.softmax(preds)
        loss = -(y * torch.log(yhat)).sum(-1).mean()

        # Weight regularization for all transformer parameters
        # for p in self.parameters():
        #     loss += self.weight_decay * (p**2).sum()

        # Compute decoding error
        pred_pos = self.place_cells.get_nearest_cell_pos(preds)
        err = torch.sqrt(((pos - pred_pos)**2).sum(-1)).mean()

        return loss, err