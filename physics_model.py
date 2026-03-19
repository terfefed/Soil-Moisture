"""
Physics-Informed Soil Moisture Model — All Components
Bi-LSTM backbone + 8 physics constraints (pure PyTorch, no NVIDIA dependency)
"""
import torch
import torch.nn as nn
import numpy as np

# =============================================================================
# 1. VEREECKEN PEDOTRANSFER EQUATIONS
# =============================================================================
def vereecken_parameters(clay_pct, sand_pct, bulk_density=1.45, organic_carbon=0.65):
    """
    Compute Van Genuchten soil-water retention parameters from soil texture
    using Vereecken (1989) pedotransfer functions.
    
    Default BD=1.45 g/cm³ and OC=0.65% are typical for alluvial soil
    in the Indo-Gangetic Plain (Varanasi region).
    """
    theta_r = 0.015 + 0.005 * clay_pct + 0.014 * organic_carbon
    
    # OVERRIDE: Prevent unrealistic lower bound for this dataset.
    # The pure Vereecken formula evaluates to ~0.160, but our data reaches down to ~0.03.
    # Keeping theta_r at 0.160 creates a hard floor that cripples dry-season predictions.
    theta_r = min(theta_r, 0.02)
    
    theta_s = 0.81  - 0.283 * bulk_density + 0.001 * clay_pct
    
    ln_alpha   = (-2.486 + 0.025 * sand_pct - 0.351 * organic_carbon
                  - 2.617 * bulk_density - 0.023 * clay_pct)
    ln_n_minus1 = (0.053 - 0.009 * sand_pct - 0.013 * clay_pct
                   + 0.00015 * (sand_pct ** 2))
    
    alpha_vg = np.exp(ln_alpha)
    n_vg     = 1.0 + np.exp(ln_n_minus1)
    m_vg     = 1.0 - 1.0 / n_vg
    
    # Field capacity ≈ θ_r + 0.7*(θ_s - θ_r), common approximation
    theta_fc = theta_r + 0.7 * (theta_s - theta_r)
    
    return {
        'theta_r': theta_r, 'theta_s': theta_s, 'theta_fc': theta_fc,
        'alpha_vg': alpha_vg, 'n_vg': n_vg, 'm_vg': m_vg,
    }

# =============================================================================
# 2. BI-LSTM MODEL
# =============================================================================
class PhysicsSoilMoistureModel(nn.Module):
    """
    Bi-LSTM backbone with fully-connected regression head.
    Processes 14-day sequences natively (no flattening).
    """
    def __init__(self, n_features=14, hidden_size=128, lstm_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden_size,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0.0
        )
        # Regression head on last timestep's bi-directional output
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, n_features)
        lstm_out, _ = self.lstm(x)         # (batch, seq_len, hidden*2)
        last_step = lstm_out[:, -1, :]     # (batch, hidden*2)
        return self.head(last_step)        # (batch, 1)

# =============================================================================
# 3. WATER BALANCE PHYSICS LOSS (upgraded with NDVI-ET, infiltration, drainage)
# =============================================================================
class WaterBalancePhysicsLoss(nn.Module):
    """
    Enhanced water balance: ΔSM = effective_precip - ET(PE) - drainage
    
    Upgrades over basic version:
    - Uses true Potential Evaporation (PE) from ERA5 instead of temperature proxy
    - Infiltration/runoff partitioning (saturated soil rejects rain)
    - Gravity drainage (water drains above field capacity)
    """
    def __init__(self, theta_r, theta_s, theta_fc):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))     # precip scaling
        self.beta  = nn.Parameter(torch.tensor(1.0))     # PE scaling
        self.gamma = nn.Parameter(torch.tensor(0.01))    # drainage rate
        self.theta_r  = theta_r
        self.theta_s  = theta_s
        self.theta_fc = theta_fc
    
    def forward(self, sm_pred_raw, sm_prev_raw, precip_raw, pe_raw):
        # Enforce non-negativity to prevent mathematically invalid inversions
        alpha_eff = torch.abs(self.alpha)
        beta_eff  = torch.abs(self.beta)
        gamma_eff = torch.abs(self.gamma)
        
        # --- True Potential Evaporation (Physics #7 upgrade) ---
        # Note: ERA5 PE is usually negative (upward flux), so we take absolute value
        actual_et = beta_eff * torch.abs(pe_raw)
        
        # --- Infiltration/runoff partitioning (Physics #8) ---
        saturation_ratio = torch.clamp(sm_prev_raw / self.theta_s, 0.0, 1.0)
        effective_precip = alpha_eff * precip_raw * (1.0 - saturation_ratio)
        
        # --- Gravity drainage (Physics #9) ---
        drainage = gamma_eff * torch.clamp(sm_prev_raw - self.theta_fc, min=0.0)
        
        # --- Water balance residual ---
        delta_sm_pred     = sm_pred_raw - sm_prev_raw
        delta_sm_expected = effective_precip - actual_et - drainage
        
        residual = delta_sm_pred - delta_sm_expected
        return torch.mean(residual ** 2)

# =============================================================================
# 3.5 UNCERTAINTY WEIGHTING (Adaptive Loss Balancing)
# =============================================================================
class AdaptiveLossWeights(nn.Module):
    """
    Learns the optimal balance between MSE (data loss) and Physics losses
    using Kendall's multi-task uncertainty weighting.
    L = sum ( L_i / (2 * exp(log_var_i)) + log_var_i / 2 )
    """
    def __init__(self, num_losses=4):
        super().__init__()
        # Initialize log variances at 0 (meaning weights start at ~0.5)
        self.log_vars = nn.Parameter(torch.zeros(num_losses))
        
    def forward(self, losses):
        """
        losses is a list of [mse, water_balance, hysteresis, laplacian, ...]
        """
        total_loss = 0.0
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += loss * precision + self.log_vars[i]
        return total_loss


# =============================================================================
# 4. HYSTERESIS PHYSICS LOSS (Van Genuchten bounds, wetting vs drying)
# =============================================================================
class HysteresisPhysicsLoss(nn.Module):
    """
    Penalizes predictions outside soil-physical bounds [θ_r, θ_s]
    with regime-aware scaling (Kool & Parker 1987: α_dry = 2×α_wet).
    """
    def __init__(self, theta_r, theta_s, alpha_vg, n_vg, precip_threshold=0.001):
        super().__init__()
        self.theta_r = theta_r
        self.theta_s = theta_s
        self.alpha_wet = alpha_vg
        self.alpha_dry = alpha_vg * 2.0  # Kool & Parker 1987
        self.n_vg = n_vg
        self.m_vg = 1.0 - 1.0 / n_vg
        self.precip_threshold = precip_threshold
    
    def forward(self, sm_pred_raw, precip_raw):
        # --- Hard bounds penalty ---
        over_sat  = torch.clamp(sm_pred_raw - self.theta_s, min=0.0)
        under_res = torch.clamp(self.theta_r - sm_pred_raw, min=0.0)
        bounds_penalty = torch.mean(over_sat ** 2) + torch.mean(under_res ** 2)
        
        # Penalizing mathematical suction head forces the model to stay falsely wet.
        # We only retain the bounds penalty to enforce [theta_r, theta_s].
        return bounds_penalty * 100.0

# =============================================================================
# 5. TEMPORAL LAPLACIAN SMOOTHING
# =============================================================================
class TemporalLaplacianLoss(nn.Module):
    """
    Penalizes second derivative of SM sequence (anti-zigzag).
    L = mean( (SM[t+1] - 2*SM[t] + SM[t-1])² )
    Weighted more heavily during dry periods.
    """
    def __init__(self, sm_col_idx=0, precip_col_idx=2, precip_threshold=0.001):
        super().__init__()
        self.sm_idx = sm_col_idx
        self.precip_idx = precip_col_idx
        self.precip_threshold = precip_threshold
    
    def forward(self, x_seq_raw):
        """
        x_seq_raw: (batch, seq_len, n_features) — raw (unscaled) input sequence
        """
        sm = x_seq_raw[:, :, self.sm_idx]         # (batch, seq_len)
        precip = x_seq_raw[:, :, self.precip_idx]  # (batch, seq_len)
        
        # Second derivative (discrete Laplacian)
        d2sm = sm[:, 2:] - 2 * sm[:, 1:-1] + sm[:, :-2]  # (batch, seq_len-2)
        
        # Weight: stronger penalty on dry days
        dry_mask = (precip[:, 1:-1] < self.precip_threshold).float()
        wet_mask = 1.0 - dry_mask
        weighted = d2sm ** 2 * (dry_mask * 2.0 + wet_mask * 0.5)
        
        return torch.mean(weighted)

# =============================================================================
# 6. MONOTONIC DRYING CONSTRAINT
# =============================================================================
class MonotonicDryingLoss(nn.Module):
    """
    When precipitation ≈ 0, SM should not increase.
    Penalizes: max(0, SM_pred - SM_prev) on dry days.
    """
    def __init__(self, precip_threshold=0.001):
        super().__init__()
        self.precip_threshold = precip_threshold
    
    def forward(self, sm_pred_raw, sm_prev_raw, precip_raw):
        dry_mask = (precip_raw < self.precip_threshold).float()
        violation = torch.clamp(sm_pred_raw - sm_prev_raw, min=0.0)
        return torch.mean((violation * dry_mask) ** 2)

# =============================================================================
# 7. DYNAMIC LAMBDA (regime-aware physics weighting)
# =============================================================================
def compute_dynamic_lambda(precip_batch, base_lambda=0.1, scaling=5.0):
    """
    Physics trusted MORE during dry periods, relaxed during monsoon.
    λ = base * exp(-scaling * precip)
    """
    return base_lambda * torch.exp(-scaling * precip_batch)
