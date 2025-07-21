import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .makemoretokens import Transformer

# -----------------------------------------------------------------------------
# Assumes you already have a `Transformer` implementation and a corresponding
# `ModelConfig` in `models.py` (the same ones you use for PatternBoost / SFT).
# This DecisionTransformer class simply *wraps* that Transformer and adds:
#   • an RTG (return‑to‑go) embedding
#   • an optional value head for actor‑critic style RL
#   • a slightly different forward signature to consume (tokens, rtg)
# -----------------------------------------------------------------------------
class DecisionTransformer(nn.Module):
    def __init__(self, vocab_size: int, config, rtg_embed_dim: int = 16, value_head: bool = True):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = config.n_embd
        self.block_size = config.block_size
        
        # RTG embedding
        self.rtg_pre_emb = nn.Linear(1, rtg_embed_dim)
        self.rtg_proj = nn.Linear(rtg_embed_dim, self.n_embd)
        
        # Core transformer (use the existing one)
        self.transformer = Transformer(config)
        
        # Value head
        self.value_head = nn.Linear(self.n_embd, 1) if value_head else None
        
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tokens, rtg, targets=None, rewards=None):
        B, T = tokens.shape
        
        # Get standard transformer embeddings
        device = tokens.device
        pos = torch.arange(0, T, dtype=torch.long, device=device).unsqueeze(0)
        tok_emb = self.transformer.transformer.wte(tokens)
        pos_emb = self.transformer.transformer.wpe(pos)
        
        # Add RTG embedding
        rtg_emb = self.rtg_proj(self.rtg_pre_emb(rtg.unsqueeze(-1)))
        
        # Combine all embeddings
        x = tok_emb + pos_emb + rtg_emb
        
        # Pass through transformer blocks
        for block in self.transformer.transformer.h:
            x = block(x)
        x = self.transformer.transformer.ln_f(x)
        
        # Output heads
        logits = self.transformer.lm_head(x)
        value = self.value_head(x).squeeze(-1) if self.value_head is not None else None
        
        # Loss calculation
        loss = None
        if targets is not None:
            if rewards is not None:
                ce_loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    reduction="none",
                    ignore_index=-1,
                )
                ce_loss = ce_loss.view(B, T)
                valid_mask = (targets != -1).float()
                weighted_loss = ce_loss * rewards * valid_mask
                reward_sum = (rewards * valid_mask).sum() + 1e-8
                loss = weighted_loss.sum() / reward_sum
            else:
                loss = F.cross_entropy(
                    logits.view(-1, self.vocab_size),
                    targets.view(-1),
                    ignore_index=-1,
                )
        
        return logits, value, loss

    @torch.no_grad()
    def generate(self, tokens, rtg, max_new_tokens: int = 1):
        for _ in range(max_new_tokens):
            # Crop context if necessary
            tokens_cond = tokens[:, -self.block_size:]
            rtg_cond = rtg[:, -self.block_size:]
            
            logits, *_ = self(tokens_cond, rtg_cond)
            next_token_logits = logits[:, -1, :]  # last step
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            tokens = torch.cat([tokens, next_token], dim=1)
            
            # Update RTG: decrease by 1 for each placed point
            next_rtg = torch.clamp(rtg[:, -1:] - 1, min=0)
            rtg = torch.cat([rtg, next_rtg], dim=1)
        
        return tokens