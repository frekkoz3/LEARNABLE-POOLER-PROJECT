"""
    This material is develop for academic purpose. 
    It is develop by Francesco Bredariol as final project of the Introduction to ML course (year 2024-2025).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MixingPooling2d(nn.Module):
    def __init__(self, kernel_size=2, stride=2, in_channels = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

        # Learning mixing coefficient per channel
        self.alpha = nn.Parameter(torch.zeros(in_channels))

    def forward(self, x):
        # Compute average and max pooling
        avg_pooled = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        max_pooled = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        # Sigmoid to let the alpha between in [0, 1]
        alpha = torch.sigmoid(self.alpha).view(1, -1, 1, 1)
        # Blend max and average pooling using the gate
        return alpha * max_pooled + (1 - alpha) * avg_pooled
    
    def get_core(self):
        """
            Return the alpha parameter.
        """
        return F.sigmoid(self.alpha)

class GatedPooling2d(nn.Module):
    def __init__(self, kernel_size=2, stride = 2, in_channels = 1):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride # same size in order to reduce the dimension

        # Gating mechanism: a sigmoid on the output of a conv2d
        self.gate_conv = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            stride = self.stride,
            groups=in_channels,  # This is the most flexibile one
            bias=True
        )

    def forward(self, x):
        # The gate is a sigmoid applied on the convolutional gate, which is a learnable conv2d
        gate = torch.sigmoid(self.gate_conv(x))  # (B, C, H, W)
        
        # Compute average and max pooling
        avg_pool = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        max_pool = F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)

        # Blend max and average pooling using the gate
        blend = (gate * max_pool + (1 - gate) * avg_pool)
        return blend

    def get_core(self):
        """
            Return te weights associated to the convolutional gate.
        """
        return self.gate_conv.weight

class StochasticPooling2d(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2, in_channels = 1, mode = "other"):
        """
            This pooler works well only with full-positive activation function.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        if mode.lower == "softmax":
            self.prob_distr = lambda x : F.softmax(x, dim = -1)
        elif mode.lower == "log_softmax":
            self.prob_distr = lambda x : F.log_softmax(x, dim = -1)
        else:
            self.prob_distr = lambda x : F.normalize(x, p = 1, dim = -1)

    def _normalize_probs(self, probs):
        # probs are coming with shape: (B, C, H_out, W_out, k*k)
        sum_probs = probs.sum(dim=-1, keepdim=True)  # shape (..., 1)
        
        # Create a mask where all elements are zero
        zero_mask = sum_probs == 0
        
        # Uniform fallback
        uniform_probs = torch.full_like(probs, 1.0 / probs.size(-1))
        
        # Where sum_probs == 0, use uniform; else use original probs
        fixed_probs = torch.where(zero_mask, uniform_probs, probs)
        return fixed_probs
    
    def _probabilistic_pool(self, x_unf):
        # Compute probabilities over the pooling window
        probs = self.prob_distr(x_unf)
        probs = self._normalize_probs(probs)
        # Sample from the distribution
        idx = torch.multinomial(probs.view(-1, self.kernel_size * self.kernel_size), 1).squeeze(-1)  # Shape: (B * C * H_out * W_out)
        # Gather selected elements
        gathered = torch.gather(
            x_unf.view(-1, self.kernel_size * self.kernel_size), 1, idx.unsqueeze(-1)
        )
        return gathered
    
    def _deterministic_pool(self, x_unf):
        # x_unf: (B, C, H_out, W_out, k*k)
        probs = self.prob_distr(x_unf)
        probs = self._normalize_probs(probs)
        expected = (probs * x_unf).sum(dim=-1)  # Expectation
        return expected

    def forward(self, x):
        B, C, H, W = x.size()
        x_unf = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # shape: (B, C, H_out, W_out, k, k)
        H_out, W_out = x_unf.size(2), x_unf.size(3)
        x_unf = x_unf.contiguous().view(B, C, H_out, W_out, -1) # shape: (B, C, H_out, W_out, k*k)
        # during training we need to do the probabilistic pooling while 
        # during inference we need to do a deterministic pooling weighted by the magnitude of the elements
        if self.training:
            return self._probabilistic_pool(x_unf).view(B, C, H_out, W_out) # need to be a little reshaped
        else:
            #return self._probabilistic_pool(x_unf).view(B, C, H_out, W_out) # if wanted to try
            return self._deterministic_pool(x_unf) # already of the right shape

    def get_core(self):
        """
            Return None since it is not a learnable pooler.
        """
        return None
    
class SP3Pooling2d(nn.Module):
    def __init__(self, kernel_size = 2, stride = 2, in_channels = 1, grid_size = 4, device = "cuda"):
        """
            This is an implementation of the SP3pooling method presented in the pdf in the folder.
            If you ask me, this is a bad implementation and does not work really fast
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.device = device
        if grid_size < stride:
            print(f"\033[93m[WARNING] : Grid size too small. The grid size must be at leas as the stride. Grid size set to {self.stride}.\033[0m")
            self.grid_size = stride
        else:
            self.grid_size = grid_size

    def _probabilistic_pool(self, x):

        x = F.adaptive_max_pool2d(x, (x.size()[2], x.size()[3])) # this is done to always obtain the same size
        
        B, C, H, W = x.size()
        # this are the blocks for the grid
        ps = [p for p in range(0, math.ceil(H/self.grid_size))]
        qs = [q for q in range(0, math.ceil(W/self.grid_size))]
        # this is the number of rows and columns to sample per block
        m = self.grid_size//self.stride
        # these are the indexes for each block, first for the rows, second for the columns
        r_p, _ = torch.sort(torch.cat([torch.multinomial(torch.ones(self.grid_size), m, replacement=False) + p*self.grid_size for p in ps], dim = 0))
        c_q, _ = torch.sort(torch.cat([torch.multinomial(torch.ones(self.grid_size), m, replacement=False) + q*self.grid_size for q in qs], dim = 0))
        r_p = torch.clamp(r_p, 0, H-1)
        c_q = torch.clamp(c_q, 0, W-1)

        r_idx = r_p.view(-1, 1) * H  
        c_idx = c_q.view(1, -1)     
        idxs = (r_idx + c_idx).flatten().to(self.device)

        gathered = torch.gather(x.view(B, C, H*W), 2, idxs.expand(B, C, -1).detach().clone())
        # seems like it works for now, so let's keep it there
        self.H_out = len(r_p)
        self.W_out = len(c_q)

        return gathered.view(B, C, self.H_out, self.W_out)

    def _deterministic_pool(self, x):
        return F.adaptive_max_pool2d(x, (self.H_out, self.W_out))

    def forward(self, x):
        """
            If for the training everything has a lot of sense, 
            the evaluating mode is a bit different.
            It seems to be a bit confused but I do not further explore it.
        """
        if self.training:
            return self._probabilistic_pool(x) 
        else:
            #return self._probabilistic_pool(x)
            return self._deterministic_pool(x) # this is the same used by Zhai in his paper
    
    def get_core(self):
        """
            Return None since it is not a learnable pooler.
        """
        return None
    
class TDGSPooling2d(nn.Module):
    """
        TDGS stands for "Temperature Driven Gumbel SoftMax". 
        In this novel pooling operator the idea is to let a learnable temperature parameter to model the softmax distribution.
        When this parameter is low (<1) the softmax produces sharpened distribution centered in the max value (similarly to a MaxPool)
        while when this parameter is high (>1) the softmax produces softer distribution (similarly an uniform StochasticPool).
        The temperature parameter serves as balancer between deterministic behavior (which is somehow related to exploitation, MaxPool)
        and stochastic behavior (which is somehow related to exploration, Uniform StochasticPool).
        Developing a parameter per pooling patch (no temperature is shared) can lead the architecture to learn
        specific spatial location where one strategy is better than the other.
    """
    def __init__(self, kernel_size = 2, stride = 2, in_channels = 1, device = "cuda", H_out = 4, W_out = 4, initial_value = 1):
        """
            This pooler works well only with full-positive activation function.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.in_channels = in_channels
        self.device = device
        rumor = (torch.rand((in_channels, H_out, W_out), device = self.device) - torch.ones((self.in_channels, H_out, W_out), device=self.device)*0.5) # some rumor from U(-0.5, 0.5)
        self.temperature = nn.Parameter(torch.ones((self.in_channels, H_out, W_out), device= self.device)*initial_value + rumor, requires_grad = True)
        self.epsilon = nn.Parameter(torch.ones_like(self.temperature, device=self.device)*10e-2, requires_grad=False) # This prevent the temperature to actually going to 0, preventing numerical errors while performing like a softmax in the bottom case
    
    def _probabilistic_pool(self, x_unf):
        # x_unf: (B, C, H_out, W_out, k*k)
        # Compute probabilities over the pooling window
        temps = F.relu(self.temperature) + self.epsilon # imposed to be >= epsilon
        # Sample from the distribution and obtain the corrispective one hot encoding 
        samples = F.gumbel_softmax(logits = x_unf/temps[None, :, :, :, None], tau = 1,  hard=True, dim = -1)
        # Since this is a one hot encoding we can do this trick to obtain the actual pooling
        pooled = (x_unf * samples).sum(dim=-1)
        return pooled
    
    def _deterministic_pool(self, x_unf):
        # x_unf: (B, C, H_out, W_out, k*k)
        temps = F.relu(self.temperature) + self.epsilon # imposed to be >= epsilon
        probs = F.gumbel_softmax(logits = x_unf/temps[None, :, :, :, None], tau = 1, dim = -1)
        expected = (probs * x_unf).sum(dim=-1)  # Expectation not just one sample
        return expected

    def forward(self, x):
        B, C, H, W = x.size()
        x_unf = x.unfold(2, self.kernel_size, self.stride).unfold(3, self.kernel_size, self.stride)  # shape: (B, C, H_out, W_out, k, k)
        H_out, W_out = x_unf.size(2), x_unf.size(3)
        x_unf = x_unf.contiguous().view(B, C, H_out, W_out, -1) # shape: (B, C, H_out, W_out, k*k)
        # during training we need to do the probabilistic pooling while 
        # during inference we need to do a deterministic pooling weighted by the magnitude of the elements
        # the inference pooling is proposed in the paper of stochastic pooling
        if self.training:
            return self._probabilistic_pool(x_unf).view(B, C, H_out, W_out) # need to be a little reshaped
        else:
            return self._probabilistic_pool(x_unf).view(B, C, H_out, W_out) # if wanted to try
            return self._deterministic_pool(x_unf) # already of the right shape 

    def get_core(self):
        """
            Return None since it is not a learnable pooler.
        """
        return self.temperature
    
if __name__ == "__main__":
    """
    b = 2
    c = 1
    t = torch.rand(b, c, 2, 2)
    print(t)
    print(torch.gather(t, 2, torch.tensor([[[[1, 1], [1, 0]] for _ in range (c)] for _ in range (b)])))
    """
    m = torch.tensor([[[[1., 2., 3., 4.], [5., 6., 7., 8.], [9., 10., 11., 12.], [13., 14., 15., 16.]]]])
    pooler = TDGSPooling2d(2, 2, 1, device="cpu")
    for i in range(10):
        print(pooler.forward(m))

