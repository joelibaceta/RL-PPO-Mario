import torch
import torch.nn as nn
import torch.nn.functional as F

class MarioConvLSTM(nn.Module):
    def __init__(self, obs_shape, n_actions):
        super().__init__()
        c, h, w = obs_shape  # obs_shape = (stack*C, H, W)
        print(f"📺 ConvLSTM Input Shape recibido: {obs_shape}")

        # CNN feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(c, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )

        # Attention module (generates attention map)
        self.attention = nn.Conv2d(64, 1, kernel_size=1)  # (batch, 1, H, W)

        # Flatten for LSTM
        self.flatten = nn.Flatten()

        # Compute flattened dim dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            features = self.feature_extractor(dummy)
            flat_dim = self.flatten(features).shape[1]
        print(f"🌟 Flat dim: {flat_dim}")

        # LSTM to handle temporal dependencies
        self.lstm = nn.LSTM(input_size=flat_dim, hidden_size=512, batch_first=True)

        # Policy and Value heads
        self.policy = nn.Linear(512, n_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x, hidden_state=None, return_attention=False):
        """
        x: (batch, seq_len, c, h, w) or (batch, c, h, w)
        hidden_state: (h_0, c_0) tuple for LSTM
        return_attention: if True, return attention maps for visualization
        """
        if x.dim() == 4:
            x = x.unsqueeze(1)  # Add temporal dim: (batch, 1, c, h, w)

        batch_size, seq_len, c, h, w = x.size()
        features_seq = []
        attn_maps_seq = []

        for t in range(seq_len):
            frame = x[:, t, :, :, :]  # (batch, c, h, w)

            # Pass through CNN
            features = self.feature_extractor(frame)  # (batch, channels, H, W)

            # Attention: generate attention map
            raw_attn_map = self.attention(features)       # (batch, 1, H, W)
            attn_map = torch.softmax(raw_attn_map.view(batch_size, -1), dim=-1)
            attn_map = attn_map.view_as(raw_attn_map)     # Reshape back to (batch, 1, H, W)

            # Apply attention (residual)
            features = features * attn_map + features

            # Flatten features for LSTM
            flat_features = self.flatten(features)    # (batch, flat_dim)
            features_seq.append(flat_features.unsqueeze(1))  # Add temporal dim
            attn_maps_seq.append(attn_map)            # Save attention map

        features_seq = torch.cat(features_seq, dim=1)  # (batch, seq_len, flat_dim)

        # Pass through LSTM
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(features_seq)
        else:
            lstm_out, hidden_state = self.lstm(features_seq, hidden_state)

        # Use last LSTM output for policy and value
        logits = self.policy(lstm_out[:, -1, :])  # (batch, n_actions)
        value = self.value(lstm_out[:, -1, :]).squeeze(-1)  # (batch,)

        if return_attention:
            return logits, value, hidden_state, attn_maps_seq
        else:
            return logits, value, hidden_state