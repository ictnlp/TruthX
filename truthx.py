import torch
from torch import nn
import torch.nn.functional as F
from abc import abstractmethod
from torch import tensor as Tensor
from typing import List, Any


class BaseVAE(nn.Module):

    def __init__(self) -> None:
        super(BaseVAE, self).__init__()

    def encode(self, input: Tensor) -> List[Tensor]:
        raise NotImplementedError

    def decode(self, input: Tensor) -> Any:
        raise NotImplementedError

    def sample(self, batch_size: int, current_device: int, **kwargs) -> Tensor:
        raise NotImplementedError

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Any, **kwargs) -> Tensor:
        pass


class MLPAE(BaseVAE):
    def __init__(
        self,
        in_channels: int,
        semantic_latent_dim: int,
        truthful_latent_dim: int,
        semantic_hidden_dims: List = None,
        truthful_hidden_dims: List = None,
        decoder_hidden_dims: List = None,
        **kwargs
    ) -> None:
        super(MLPAE, self).__init__()

        self.semantic_latent_dim = semantic_latent_dim

        if semantic_hidden_dims is None:
            semantic_hidden_dims = []

        # Build Semantic Encoder
        semantic_encoder_modules = []
        flat_size = in_channels
        for h_dim in semantic_hidden_dims:
            semantic_encoder_modules.append(
                nn.Sequential(
                    nn.Linear(flat_size, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU()
                )
            )
            flat_size = h_dim
        semantic_encoder_modules.append(
            nn.Sequential(
                nn.Linear(flat_size, semantic_latent_dim),
                nn.LayerNorm(semantic_latent_dim),
                nn.LeakyReLU(),
            )
        )

        self.semantic_encoder = nn.Sequential(*semantic_encoder_modules)

        if truthful_hidden_dims is None:
            truthful_hidden_dims = []

        # Build Truthful Encoder
        truthful_encoder_modules = []
        flat_size = in_channels
        for h_dim in truthful_hidden_dims:
            truthful_encoder_modules.append(
                nn.Sequential(
                    (
                        nn.Linear(flat_size, h_dim)
                        if flat_size != h_dim
                        else nn.Identity()
                    ),
                    nn.LayerNorm(h_dim),
                    nn.LeakyReLU(),
                )
            )
            flat_size = h_dim
        truthful_encoder_modules.append(
            nn.Sequential(
                (
                    nn.Linear(flat_size, truthful_latent_dim)
                    if flat_size != truthful_latent_dim
                    else nn.Identity()
                ),
                nn.LayerNorm(truthful_latent_dim),
                nn.LeakyReLU(),
            )
        )

        self.truthful_encoder = nn.Sequential(*truthful_encoder_modules)

        # Cross-Attention Module
        self.num_heads = 1
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=semantic_latent_dim, num_heads=self.num_heads
        )

        self.proj = None
        if semantic_latent_dim != truthful_latent_dim:
            self.proj = nn.Linear(truthful_latent_dim, semantic_latent_dim, bias=False)

        # Build Decoder
        decoder_modules = []
        if len(decoder_hidden_dims) > 0:
            flat_size = semantic_latent_dim
            for h_dim in decoder_hidden_dims:
                decoder_modules.append(
                    nn.Sequential(
                        nn.Linear(flat_size, h_dim), nn.LayerNorm(h_dim), nn.LeakyReLU()
                    )
                )
                flat_size = h_dim

            flat_size = decoder_hidden_dims[-1]
            self.decoder = nn.Sequential(*decoder_modules)
        else:
            self.decoder_input = None

            self.decoder = None
            flat_size = semantic_latent_dim
        self.final_layer = nn.Sequential(nn.Linear(flat_size, in_channels))

    def encode_semantic(self, input: Tensor) -> List[Tensor]:
        semantic_latent_rep = self.semantic_encoder(input)
        return semantic_latent_rep

    def encode_truthful(self, input: Tensor) -> List[Tensor]:
        truthful_latent_rep = self.truthful_encoder(input)
        truthful_latent_rep = F.normalize(truthful_latent_rep, p=2, dim=-1)

        return truthful_latent_rep

    def attention(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        if self.proj is not None and query.size(-1) != key.size(-1):
            key = self.proj(key)
            value = self.proj(value)
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)

        output, attention_weights = self.cross_attention(query, key, value)

        return output[0]

    def decode(self, z: Tensor) -> Tensor:
        result = z
        if self.decoder is not None:
            result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def forward(
        self, input: Tensor, truthful_latent_rep=None, **kwargs
    ) -> List[Tensor]:
        semantic_latent_rep = self.encode_semantic(input)
        if truthful_latent_rep is None:
            truthful_latent_rep = self.encode_truthful(input)
        truthful_latent_rep = truthful_latent_rep.reshape(
            -1, truthful_latent_rep.size(-1)
        )
        z = semantic_latent_rep + self.attention(
            semantic_latent_rep,
            truthful_latent_rep.contiguous(),
            truthful_latent_rep.contiguous(),
        )
        output = self.decode(z)

        return [output, input, semantic_latent_rep, truthful_latent_rep]

    def forward_decoder(self, input, semantic_latent_rep, truthful_latent_rep):
        z = semantic_latent_rep + self.attention(
            semantic_latent_rep, truthful_latent_rep, truthful_latent_rep
        )
        output = self.decode(z)
        return [output, input, semantic_latent_rep, truthful_latent_rep]

    def get_semantic_latent_rep(self, input: Tensor, **kwargs) -> List[Tensor]:
        semantic_latent_rep = self.encode_semantic(input)
        return semantic_latent_rep

    def get_truthful_latent_rep(self, input: Tensor, **kwargs) -> List[Tensor]:
        truthful_latent_rep = self.encode_truthful(input)
        return truthful_latent_rep

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        recons_loss = F.mse_loss(recons, input)

        loss = recons_loss
        return {"loss": loss, "Reconstruction_Loss": recons_loss.detach()}


class TruthX:
    def __init__(self, model_path, hidden_size, edit_strength=1.0, top_layers=10):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path)
        args = checkpoint["args"]

        semantic_latent_dim = args.semantic_latent_dim  # Adjust as needed
        truthful_latent_dim = args.truthful_latent_dim
        semantic_hidden_dims = (
            [int(_) for _ in args.semantic_hidden_dims.split(",")]
            if args.semantic_hidden_dims != ""
            else []
        )
        truthful_hidden_dims = (
            [int(_) for _ in args.truthful_hidden_dims.split(",")]
            if args.truthful_hidden_dims != ""
            else []
        )
        decoder_hidden_dims = (
            [int(_) for _ in args.decoder_hidden_dims.split(",")]
            if args.decoder_hidden_dims != ""
            else []
        )

        ae_model = MLPAE(
            in_channels=hidden_size,
            semantic_latent_dim=semantic_latent_dim,
            truthful_latent_dim=truthful_latent_dim,
            semantic_hidden_dims=semantic_hidden_dims,
            truthful_hidden_dims=truthful_hidden_dims,
            decoder_hidden_dims=decoder_hidden_dims,
        ).to(device)

        ae_model.load_state_dict(checkpoint["state_dict"])

        ae_model.pos_center = ((checkpoint["pos_center"])).to(device)
        ae_model.neg_center = ((checkpoint["neg_center"])).to(device)
        ae_model.eval()
        ae_model.to(device)
        self.ae_model = ae_model
        # checkpoint['accuracy'][-1]=1.0

        self.rank = checkpoint["rank"]

        self.top_layers = top_layers
        self.edit_strength = edit_strength
        self.cur_layer_id = 0
        self.prompt_length = None
        self.mc = False

    @torch.inference_mode()
    def edit(self, X):
        layer_id = int(self.cur_layer_id.split(".")[0])
        if self.cur_layer_id.endswith("attn"):
            layer_id = 2 * layer_id
        else:
            layer_id = 2 * layer_id + 1

        if self.rank[layer_id] > self.top_layers:
            return X

        bsz, s_len, d = X.size()
        x = (
            X.contiguous()
            .view(-1, d)
            .type_as(self.ae_model.semantic_encoder[0][0].weight)
        )
        x_truthful = self.ae_model.get_truthful_latent_rep(
            X.type_as(self.ae_model.semantic_encoder[0][0].weight)
        )

        pos_center = self.ae_model.pos_center[layer_id].unsqueeze(0)
        neg_center = self.ae_model.neg_center[layer_id].unsqueeze(0)

        delta = (pos_center - neg_center).unsqueeze(0)
        recon_x_pos = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful + delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d)
        )
        recon_x_neg = (
            self.ae_model(
                x,
                truthful_latent_rep=F.normalize(
                    x_truthful - delta, p=2, dim=-1
                ).type_as(x),
            )[0]
            .contiguous()
            .view(bsz, s_len, d)
        )
        Delta = recon_x_pos - recon_x_neg
        Delta = Delta.contiguous().to(X.dtype)
        Delta = F.normalize(Delta, p=2, dim=-1).type_as(X) * torch.norm(
            X, p=2, dim=-1
        ).unsqueeze(2)

        mask = torch.ones((bsz, s_len), device=Delta.device)

        if self.mc:
            # multiple-choice, only edit the tokens in answer
            mask[:, : self.prompt_length + 1] = 0
            # probing those untruthful position
            probing = (
                torch.nn.functional.cosine_similarity(
                    x_truthful, neg_center.unsqueeze(1), dim=-1
                )
                - torch.nn.functional.cosine_similarity(
                    x_truthful, pos_center.unsqueeze(1), dim=-1
                )
            ).clamp(0, 999)
            mask = mask * probing

        else:
            # open-ended generation, only edit the generated token (i.e., last token)
            mask[:, :-1] = 0
            mask[:, -1:] = 1

        new_X = X + (Delta.type_as(X)) * self.edit_strength * mask.unsqueeze(2).type_as(X)
        return new_X
