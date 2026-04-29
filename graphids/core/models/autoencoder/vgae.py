from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # used by neighborhood_loss_negsampled

from .._conv import (
    InputEncoder,
    build_conv_stack,
    build_encoder_stack,
    conv_forward,
)


class GraphAutoencoderNeighborhood(nn.Module):
    """Variational graph autoencoder with mask-recon + canid + nbr aux heads.

    Encoder maps node features to ``q(z|x) = N(mu, σ²)``; decoder
    reconstructs continuous features from the reparameterized ``z``.
    Mask-recon training (15% random node masking) commits the encoder
    to "predict v from neighborhood" rather than "echo v back".

    Two auxiliary training heads operate on the masked-input latent:
    - ``canid_classifier`` (Linear → num_ids): predict the masked
      node's CAN ID from its neighbor-derived ``z``.
    - ``neighborhood_decoder`` (small MLP → num_ids): predict the
      multiset of neighbor CAN IDs from ``z``.
    Both are training-only — they shape ``μ`` to encode CAN-ID
    identity and neighborhood structure (which empirically gives the
    pre-mask-recon code its 0.76 AUC on test_03 zero-day) but are
    NOT used by the test-time anomaly score, which is calibrated
    max-σ over (recon, Mahalanobis on μ, KL).

    Mask signaling has two parts: (a) a frozen zero-init Parameter at
    the input layer that replaces ``x[v]`` for masked nodes, and (b) a
    reserved ``mask_id`` slot in the id_encoder vocab. Both must hide
    — masking only continuous bytes leaves the encoder access to v's
    identity. ``mask_id = num_ids`` and the id_encoder is sized to
    ``num_ids + 1`` (set up by ``VGAEModule._build``).

    See ``~/plans/vgae-mask-recon-and-latent-density.md`` (synthesis)
    and the integration of pre-synthesis canid/nbr aux heads (see git
    log on this file).
    """

    def __init__(
        self,
        id_encoder,
        num_ids,
        in_channels,
        hidden_dims=None,
        latent_dim=32,
        encoder_heads=4,
        decoder_heads=4,
        dropout=0.1,
        batch_norm=True,
        use_checkpointing=False,
        conv_type="gat",
        edge_dim=None,
        proj_dim=0,
        mlp_hidden=None,
    ):
        super().__init__()

        # Shared input encoding (ID encoder + optional projection)
        self.input_encoder = InputEncoder(
            id_encoder=id_encoder,
            in_channels=in_channels,
            conv_type=conv_type,
            edge_dim=edge_dim,
            proj_dim=proj_dim,
        )
        self.dropout_rate = dropout
        self.batch_norm = batch_norm
        self.use_checkpointing = use_checkpointing
        self.conv_type = conv_type
        self._uses_edge_attr = self.input_encoder._uses_edge_attr
        self._edge_dim = self.input_encoder._edge_dim
        self._proj_dim = proj_dim
        self.in_channels = in_channels

        # Encoder conv stack (shared with DGI)
        gat_in_dim = self.input_encoder.out_dim
        self.gat_in_dim = gat_in_dim
        self.encoder_layers, self.encoder_bns, encoder_targets = build_encoder_stack(
            hidden_dims,
            latent_dim,
            gat_in_dim,
            conv_type,
            self._edge_dim,
            encoder_heads=encoder_heads,
            batch_norm=batch_norm,
        )
        self.latent_in_dim = encoder_targets[-1]
        self.z_mean = nn.Linear(self.latent_in_dim, latent_dim)
        self.z_logvar = nn.Linear(self.latent_in_dim, latent_dim)

        # Decoder: mirror of encoder, final layer outputs continuous features
        decoder_targets = list(reversed(encoder_targets))
        decoder_targets[-1] = in_channels
        self.decoder_layers, self.decoder_bns = build_conv_stack(
            conv_type,
            latent_dim,
            decoder_targets,
            self._edge_dim,
            heads_first=decoder_heads,
            batch_norm=batch_norm,
        )
        # Last decoder layer is linear (no BN)
        if batch_norm and len(self.decoder_bns) == len(decoder_targets):
            self.decoder_bns = self.decoder_bns[:-1]

        self.latent_dim = latent_dim
        self.num_ids = num_ids

        # Auxiliary training heads (NOT used at test scoring). They shape
        # μ to encode CAN-ID identity + neighborhood structure during
        # training. Inputs to both are the masked-input latent z, so each
        # is a legitimate prediction task ("recover masked node's ID /
        # neighbors from its neighbor context"), not an identity echo.
        self.canid_classifier = nn.Linear(latent_dim, num_ids)
        if mlp_hidden is None:
            mlp_hidden = latent_dim
        self.neighborhood_decoder = nn.Sequential(
            nn.Linear(latent_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_ids),
        )

        # Frozen "this node is hidden" signal at the input layer. Zero-init
        # + requires_grad=False because a learnable token would converge to
        # the empirical mean of node features, defeating the purpose of
        # masking. id_encoder side reserves slot num_ids for the same
        # signal in identity space.
        self.mask_token = nn.Parameter(torch.zeros(in_channels), requires_grad=False)
        self.mask_id: int = num_ids

    def apply_random_mask(
        self, x: torch.Tensor, node_id: torch.Tensor, mask_rate: float = 0.15
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace ``mask_rate`` fraction of (x, node_id) rows with mask token + mask_id."""
        n = x.size(0)
        mask = torch.rand(n, device=x.device) < mask_rate
        x = x.clone()
        node_id = node_id.clone()
        x[mask] = self.mask_token
        node_id[mask] = self.mask_id
        return x, node_id, mask

    @staticmethod
    def neighborhood_loss_negsampled(
        logits: torch.Tensor,
        node_id: torch.Tensor,
        edge_index: torch.Tensor,
        num_ids: int,
        k_neg: int = 32,
    ) -> torch.Tensor:
        """Neighborhood BCE loss with negative sampling.

        Memory: O(num_edges + num_nodes * k_neg) instead of O(num_nodes * num_ids).
        """
        src, dst = edge_index
        dst_ids = node_id[dst]
        valid = (dst_ids >= 0) & (dst_ids < num_ids)
        pos_logits = logits[src[valid], dst_ids[valid]]
        pos_loss = -F.logsigmoid(pos_logits).mean()
        neg_ids = torch.randint(0, num_ids, (logits.size(0), k_neg), device=logits.device)
        pos_keys = src[valid].long() * num_ids + dst_ids[valid].long()
        node_range = torch.arange(logits.size(0), device=logits.device)
        neg_keys = (node_range.unsqueeze(1) * num_ids + neg_ids).reshape(-1)
        is_collision = torch.isin(neg_keys, pos_keys)
        neg_logits = logits.gather(1, neg_ids).reshape(-1)[~is_collision]
        neg_loss = (
            -F.logsigmoid(-neg_logits).mean() if neg_logits.numel() > 0 else logits.new_zeros(1)
        )
        return pos_loss + neg_loss

    def encode(self, x, edge_index, edge_attr=None, batch=None, node_id=None):
        """Returns ``(z, kl_per_node, mu)``.

        ``kl_per_node`` is per-node KL divergence (mean over latent dims).
        Training loss takes ``.mean()`` for a scalar gradient; test-time
        scoring scatter-means to per-graph for the KL anomaly axis.
        ``mu`` is the encoder's mean output, pre-reparameterization —
        used for Mahalanobis scoring (avoids reparam noise polluting
        latent-density signal).
        """
        x = self.input_encoder(x, node_id)
        for i, conv in enumerate(self.encoder_layers):
            bn = self.encoder_bns[i] if self.batch_norm else None
            x = conv_forward(
                conv,
                x,
                edge_index,
                edge_attr,
                bn=bn,
                batch=batch,
                dropout_p=self.dropout_rate,
                training=self.training,
                use_checkpointing=self.use_checkpointing,
            )
        mu = self.z_mean(x)
        # Clamp to ±10 so exp(logvar) stays inside fp16 range (~65504) under
        # autocast — at ±20, exp(20)≈4.85e8 overflows fp16 and propagates NaN
        # through the KL term during validation.
        logvar = self.z_logvar(x).clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        kl_per_node = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean(dim=-1)
        return z, kl_per_node, mu

    def decode_node(self, z, edge_index, edge_attr=None, batch=None):
        assert z.size(-1) == self.latent_dim, (
            f"Expected {self.latent_dim}D input, got {z.size(-1)}D"
        )
        x = z

        for i, conv in enumerate(self.decoder_layers):
            if i < len(self.decoder_layers) - 1:
                bn = self.decoder_bns[i] if self.batch_norm else None
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    bn=bn,
                    batch=batch,
                    dropout_p=self.dropout_rate,
                    training=self.training,
                    use_checkpointing=self.use_checkpointing,
                )
            else:
                # Last decoder layer — linear (no terminal nonlinearity).
                # Inputs are z-score standardized to N(0,1), so output range
                # must include negatives + values >1; matches AnomalyDAE /
                # DOMINANT / GAD-NR convention for arbitrary-valued attribute
                # decoders.
                x = conv_forward(
                    conv,
                    x,
                    edge_index,
                    edge_attr,
                    activation=None,
                    use_checkpointing=self.use_checkpointing,
                )
        return x  # cont_out, shape: [num_nodes, in_channels]

    def forward(self, x, edge_index, batch, edge_attr=None, node_id=None):
        # 5-tuple: (cont_out, canid_logits, nbr_logits, z, kl_per_node).
        # Aux logits are training-only; test scoring (_score in
        # vgae_module) ignores them. Computing them on every forward
        # adds two small linear/MLP heads on z — negligible vs the
        # GAT encoder/decoder stack.
        ea = edge_attr if self._uses_edge_attr else None
        z, kl_per_node, _mu = self.encode(x, edge_index, edge_attr=ea, batch=batch, node_id=node_id)
        cont_out = self.decode_node(z, edge_index, edge_attr=ea, batch=batch)
        canid_logits = self.canid_classifier(z)
        nbr_logits = self.neighborhood_decoder(z)
        return cont_out, canid_logits, nbr_logits, z, kl_per_node
