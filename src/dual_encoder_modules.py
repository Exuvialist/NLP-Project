"""
dual_encoder_modules.py
=======================
PyTorch implementation of the Lightweight Multimodal Dual Encoder architecture:
1. ArticleAttentionPooling: Article-level daily attention pooling.
2. TabularEncoder: Residual MLP for financial time-series and news shock features.
3. IntermediateGatedFusion: Convex combination gated fusion between text and tabular modalities.
4. MultimodalDualEncoder: Complete end-to-end multimodal network.
5. Statistical significance tests: Pesaran-Timmermann and McNemar test.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class ArticleAttentionPooling(nn.Module):
    """
    Article-level daily attention pooling.
    Takes article embeddings E_t in R^(B x K x d_pca) and padding mask in {0, 1}^(B x K).
    Computes normalized attention weights alpha_(t,k) and returns daily text vector h_text.
    """
    def __init__(self, d_pca: int = 32, d_attn: int = 32):
        super().__init__()
        self.d_pca = d_pca
        self.d_attn = d_attn
        self.attn_net = nn.Sequential(
            nn.Linear(d_pca, d_attn),
            nn.Tanh(),
            nn.Linear(d_attn, 1)
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor):
        """
        Parameters:
        -----------
        embeddings: torch.Tensor of shape (B, K, d_pca)
        mask: torch.Tensor of shape (B, K), True/1 for valid articles, False/0 for padding

        Returns:
        --------
        h_text: torch.Tensor of shape (B, d_pca)
        alpha: torch.Tensor of shape (B, K) - attention weights
        """
        # Raw scores: (B, K, 1) -> (B, K)
        scores = self.attn_net(embeddings).squeeze(-1)
        
        # Mask padding: set masked elements to -1e9 before softmax
        mask_bool = mask.bool()
        scores = scores.masked_fill(~mask_bool, -1e9)
        
        # Softmax over articles
        alpha = F.softmax(scores, dim=-1) # (B, K)
        
        # Safety for days with all masked (if any): replace NaNs with zeros
        alpha = torch.nan_to_num(alpha, nan=0.0)
        
        # Weighted sum: (B, 1, K) @ (B, K, d_pca) -> (B, 1, d_pca) -> (B, d_pca)
        h_text = torch.bmm(alpha.unsqueeze(1), embeddings).squeeze(1)
        return h_text, alpha


class TabularEncoder(nn.Module):
    """
    Residual 2-Layer MLP for tabular market features (8 technical + 7 news shock = 15 features).
    """
    def __init__(self, in_dim: int = 15, hidden_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.linear1 = nn.Linear(in_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x_tab: torch.Tensor) -> torch.Tensor:
        # x_tab: (B, in_dim)
        h1 = self.relu(self.norm1(self.linear1(x_tab)))
        h1 = self.dropout(h1)
        h2 = self.linear2(h1)
        h_tab = self.relu(self.norm2(h1 + h2)) # Residual connection
        return h_tab


class IntermediateGatedFusion(nn.Module):
    """
    Intermediate Gated Fusion module.
    Projects h_text and h_tab to shared latent dimension D in {16, 32}.
    Computes adaptive gate vector g_t in (0, 1)^D conditioned on [z_text || z_tab || has_news].
    Returns z_fused = g_t * z_text + (1 - g_t) * z_tab.
    """
    def __init__(self, d_text: int = 32, d_tab: int = 32, d_latent: int = 32, dropout: float = 0.4):
        super().__init__()
        self.proj_text = nn.Sequential(
            nn.Linear(d_text, d_latent),
            nn.LayerNorm(d_latent),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        self.proj_tab = nn.Sequential(
            nn.Linear(d_tab, d_latent),
            nn.LayerNorm(d_latent),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        # Gate takes [z_text (D) + z_tab (D) + has_news (1)] = 2*D + 1
        self.gate_net = nn.Sequential(
            nn.Linear(2 * d_latent + 1, d_latent),
            nn.Sigmoid()
        )

    def forward(self, h_text: torch.Tensor, h_tab: torch.Tensor, has_news: torch.Tensor):
        z_text = self.proj_text(h_text) # (B, D)
        z_tab = self.proj_tab(h_tab)    # (B, D)
        
        # Ensure has_news has shape (B, 1)
        if has_news.dim() == 1:
            has_news = has_news.unsqueeze(1)
            
        gate_input = torch.cat([z_text, z_tab, has_news], dim=-1)
        g_t = self.gate_net(gate_input) # (B, D)
        
        z_fused = g_t * z_text + (1.0 - g_t) * z_tab
        return z_fused, g_t


class MultimodalDualEncoder(nn.Module):
    """
    Full Lightweight Multimodal Dual Encoder for daily exchange rate direction prediction.
    """
    def __init__(
        self,
        d_pca: int = 32,
        tabular_dim: int = 15,
        d_attn: int = 32,
        d_tab_hidden: int = 32,
        d_latent: int = 32,
        dropout: float = 0.4
    ):
        super().__init__()
        self.text_encoder = ArticleAttentionPooling(d_pca=d_pca, d_attn=d_attn)
        self.tab_encoder = TabularEncoder(in_dim=tabular_dim, hidden_dim=d_tab_hidden, dropout=dropout)
        self.fusion = IntermediateGatedFusion(
            d_text=d_pca,
            d_tab=d_tab_hidden,
            d_latent=d_latent,
            dropout=dropout
        )
        self.head = nn.Linear(d_latent, 1)

    def forward(
        self,
        article_embeddings: torch.Tensor,
        article_mask: torch.Tensor,
        tabular_features: torch.Tensor,
        has_news: torch.Tensor
    ):
        # 1. Attention pooling over daily articles
        h_text, alpha = self.text_encoder(article_embeddings, article_mask)
        
        # 2. Tabular residual encoding
        h_tab = self.tab_encoder(tabular_features)
        
        # 3. Gated fusion
        z_fused, g_t = self.fusion(h_text, h_tab, has_news)
        
        # 4. Binary classification logit
        logits = self.head(z_fused).squeeze(-1) # (B,)
        return logits, alpha, g_t


class UnimodalTextOnlyModel(nn.Module):
    """
    EXP-03b: Unimodal text neural model (Article Attention Pooling -> Linear head).
    """
    def __init__(self, d_pca: int = 32, d_attn: int = 32, dropout: float = 0.4):
        super().__init__()
        self.text_encoder = ArticleAttentionPooling(d_pca=d_pca, d_attn=d_attn)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_pca, 1)
        )

    def forward(self, article_embeddings: torch.Tensor, article_mask: torch.Tensor):
        h_text, alpha = self.text_encoder(article_embeddings, article_mask)
        logits = self.head(h_text).squeeze(-1)
        return logits, alpha


class UnimodalTabularModel(nn.Module):
    """
    EXP-02 (Neural): Tabular residual MLP -> Linear head.
    """
    def __init__(self, in_dim: int = 15, hidden_dim: int = 32, dropout: float = 0.4):
        super().__init__()
        self.tab_encoder = TabularEncoder(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, tabular_features: torch.Tensor):
        h_tab = self.tab_encoder(tabular_features)
        logits = self.head(h_tab).squeeze(-1)
        return logits


# ---------------------------------------------------------------------------
# Statistical Significance Tests
# ---------------------------------------------------------------------------

def pesaran_timmermann_test(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Pesaran & Timmermann (1992) test of directional accuracy in financial markets.
    H0: Predictions and real outcomes are independently distributed (no directional skill).
    Returns S_n test statistic and one-sided p-value (standard normal).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    n = len(y_true)
    
    if n == 0:
        return {"statistic": 0.0, "p_value": 1.0, "directional_accuracy": 0.0}
        
    p_hat = np.mean(y_true == y_pred)
    p_y = np.mean(y_true)
    p_x = np.mean(y_pred)
    
    p_star = p_y * p_x + (1.0 - p_y) * (1.0 - p_x)
    
    # Variance of p_hat
    v_p_hat = (1.0 / n) * p_star * (1.0 - p_star)
    
    # Variance of p_star
    v_p_star = (
        (1.0 / n) * ((2.0 * p_x - 1.0) ** 2) * p_y * (1.0 - p_y) +
        (1.0 / n) * ((2.0 * p_y - 1.0) ** 2) * p_x * (1.0 - p_x) +
        (4.0 / (n ** 2)) * p_x * p_y * (1.0 - p_x) * (1.0 - p_y)
    )
    
    denom = np.sqrt(v_p_hat - v_p_star) if v_p_hat > v_p_star else 1e-12
    s_n = (p_hat - p_star) / denom
    
    # One-sided p-value: probability of observing S_n >= calculated under H0
    p_value = 1.0 - stats.norm.cdf(s_n)
    
    return {
        "directional_accuracy": float(p_hat),
        "p_star_baseline": float(p_star),
        "statistic": float(s_n),
        "p_value": float(p_value)
    }


def mcnemar_test(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """
    McNemar's test for comparing paired categorical predictions between two models.
    b: Model A correct, Model B incorrect.
    c: Model A incorrect, Model B correct.
    Uses continuity correction.
    """
    y_true = np.asarray(y_true).astype(int)
    pred_a = np.asarray(pred_a).astype(int)
    pred_b = np.asarray(pred_b).astype(int)
    
    corr_a = (pred_a == y_true)
    corr_b = (pred_b == y_true)
    
    b = int(np.sum(corr_a & ~corr_b)) # A right, B wrong
    c = int(np.sum(~corr_a & corr_b)) # A wrong, B right
    
    total_discordant = b + c
    if total_discordant == 0:
        return {"b": b, "c": c, "chi2": 0.0, "p_value": 1.0}
        
    chi2 = ((abs(b - c) - 1.0) ** 2) / total_discordant
    p_value = 1.0 - stats.chi2.cdf(chi2, df=1)
    
    return {
        "b_model_a_superior": b,
        "c_model_b_superior": c,
        "chi2": float(chi2),
        "p_value": float(p_value)
    }
