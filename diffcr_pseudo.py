import os
import torch
import torch.nn as nn
import numpy as np
from einops import repeat
import torch.nn.functional as F

class STE_Ceil(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_in):
        x = torch.ceil(x_in)
        return x
    
    @staticmethod
    def backward(ctx, g):
        return g, None
    
ste_ceil = STE_Ceil.apply

def select_topk_and_remaining_tokens(x, token_weights, k, C):
    """
    Selects top-k and remaining tokens based on the token weights.

    Args:
        x (torch.Tensor): Input tensor of shape (B, N, C).
        token_weights (torch.Tensor): Weights tensor of shape (B, N).
        k (int): Number of top tokens to select.
        C (int): Number of channels.

    Returns:
        topk_x (torch.Tensor): Top-k tokens of shape (B, k, C).
        remaining_x (torch.Tensor): Remaining tokens of shape (B, N, C).
        topk_indices (torch.Tensor): Indices of top-k tokens of shape (B, k).
    """
    B, N, _ = x.shape
    topk_weights, topk_indices = torch.topk(torch.sigmoid(token_weights), k=k, sorted=False)
    sorted_indices, index = torch.sort(topk_indices, dim=1)

    # Get top-k tokens
    topk_x = x.gather(
        dim=1,
        index=repeat(sorted_indices, 'b t -> b t d', d=C)
    )

    # Get remaining tokens
    remaining_x = x.clone()
    remaining_x.scatter_(1, repeat(sorted_indices, 'b t -> b t d', d=C), torch.zeros_like(topk_x))

    return topk_weights, topk_x, remaining_x, sorted_indices, index

class PixArtBlock(nn.Module):

    def __init__(self, hidden_size, routing=True, mod_ratio=0, diffcr=True, mod_granularity=0.01, **block_kwargs):
        super().__init__()

        """Initialize the PixArt block (Omitted)."""

        # router
        self.routing = routing
        self.mod_ratio = mod_ratio
        if self.routing:
            self.router = nn.Linear(hidden_size, 1, bias=False)

        # learnable ratios
        self.diffcr = diffcr
        if self.diffcr:
            self.kept_ratio_candidate = nn.Parameter(torch.arange(1, 0.2, -0.1).float())
            self.kept_ratio_candidate.requires_grad_(False)
            self.diff_ratio = nn.Parameter(torch.tensor(1.0))
            self.diff_ratio.requires_grad_(True)

    def find_nearest_bins(self, kept_ratio):
        # Calculate the absolute differences between diff_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)

        # Find the indices of the two smallest differences
        _, indices = torch.topk(differences, 2, largest=False)
        
        # Get the values corresponding to these indices
        nearest_bins = self.kept_ratio_candidate[indices]
        
        return nearest_bins, indices

    def find_closest_bin(self, kept_ratio):
        # Calculate the absolute differences between kept_ratio and each candidate value
        differences = torch.abs(self.kept_ratio_candidate - kept_ratio)

        # Find the index of the smallest difference
        closest_index = torch.argmin(differences)

        # Return the closest bin value
        return self.kept_ratio_candidate[closest_index]

    def forward(self, x, y, t, mask=None, timestep=0, T=0):

        B, N, C = x.shape

        if self.routing and self.diffcr:

            if self.training:

                kept_ratio = torch.clamp(self.diff_ratio, 0.1, 1.0)
                nearest_bins, indices = self.find_nearest_bins(kept_ratio)

                lower_bin, upper_bin = nearest_bins[0], nearest_bins[1]
                lower_weight = (upper_bin - kept_ratio) / (upper_bin - lower_bin)
                upper_weight = 1.0 - lower_weight

                # lower outputs

                capacity = ste_ceil(lower_bin * N).to(torch.int32)
                k = torch.min(capacity, torch.tensor(N, device=x.device))

                token_weights = self.router(x).squeeze(2)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
                topk_x = topk_x + self.cross_attn(topk_x, y, mask)
                lower_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                lower_out = lower_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                lower_out = lower_out + topk_x

                lower_out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=lower_out
                )

                # upper outputs

                capacity = ste_ceil(upper_bin * N).to(torch.int32)
                k = torch.min(capacity, torch.tensor(N, device=x.device))

                token_weights = self.router(x).squeeze(2)
                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
                topk_x = topk_x + self.cross_attn(topk_x, y, mask)
                upper_out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                upper_out = upper_out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                upper_out = upper_out + topk_x

                upper_out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=upper_out
                )

                # Linear combination of the two outputs
                out = lower_weight * lower_out + upper_weight * upper_out

                return out, kept_ratio

            else:
                
                kept_ratio = torch.clamp(self.diff_ratio, 0.1, 1.0)
                nearest_bins, indices = self.find_nearest_bins(kept_ratio)
                kept_ratio = nearest_bins[0]

                capacity = ste_ceil(kept_ratio*N).to(torch.int32)
                k =  torch.min(capacity, torch.tensor(N, device=x.device))

                token_weights = self.router(x).squeeze(2)

                topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
                topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
                topk_x = topk_x + self.cross_attn(topk_x, y, mask)
                out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

                out = out * topk_weights.gather(dim=1, index=index).unsqueeze(2)
                out = out + topk_x

                # Combine bypassed tokens and processed topk tokens
                out = remaining_x.scatter_add(
                    dim=1,
                    index=repeat(sorted_indices, 'b t -> b t d', d=C),
                    src=out
                )
                
            return out, kept_ratio

        
        elif self.routing and not self.diffcr:

            token_weights = self.router(x).squeeze(2)

            capacity = int((1 - self.mod_ratio) * N)
            k = min(N, capacity)
            k = max(k, 1)
            topk_weights, topk_x, remaining_x, sorted_indices, index = select_topk_and_remaining_tokens(x, token_weights, k, C)

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
            topk_x = topk_x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(topk_x), shift_msa, scale_msa)).reshape(B, k, C))
            topk_x = topk_x + self.cross_attn(topk_x, y, mask)
            out = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(topk_x), shift_mlp, scale_mlp)))

            out *= topk_weights.gather(dim=1, index=index).unsqueeze(2)
            out += topk_x

            # Combine bypassed tokens and processed topk tokens
            out = remaining_x.scatter_add(
                dim=1,
                index=repeat(sorted_indices, 'b t -> b t d', d=C),
                src=out
            )

        else:

            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (self.scale_shift_table[None] + t.reshape(B, 6, -1)).chunk(6, dim=1)
            x = x + self.drop_path(gate_msa * self.attn(t2i_modulate(self.norm1(x), shift_msa, scale_msa)).reshape(B, N, C))
            x = x + self.cross_attn(x, y, mask)
            out = x + self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp)))

        return out
    

def calculate_diffcr_loss(self, ratios, target_ratio):

    avg_mod_ratio = torch.mean(torch.stack(ratios))
    target_ratio_tensor = torch.tensor([target_ratio], device=avg_mod_ratio.device)
    loss = F.mse_loss(avg_mod_ratio, target_ratio_tensor)

    return loss