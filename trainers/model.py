import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from sklearn.decomposition import PCA

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        # separate linear layers for q,k,v
        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = self.q_linear(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_linear(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_linear(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, cfg, n=12):
        super().__init__()
        self.embed_dim = int(cfg.LSN.EMBED_DIM)
        self.mlp_ratio = cfg.LSN.MLP_RATIO
        self.norm1 = nn.LayerNorm(self.embed_dim)
        n_heads = int(self.embed_dim //64)
        self.attn = Attention(self.embed_dim,
                              num_heads=n_heads,
                              qkv_bias=cfg.LSN.QKV_BIAS,
                              qk_scale=cfg.LSN.QK_SCALE,
                              attn_drop=cfg.LSN.ATTN_DROP,
                              proj_drop=cfg.LSN.DROP)
        self.norm2 = nn.LayerNorm(self.embed_dim)
        mlp_hidden_dim = int(self.embed_dim * self.mlp_ratio)
        self.mlp = Mlp(in_features=self.embed_dim, hidden_features=mlp_hidden_dim)
        self.n = n
        self.temp_layers = cfg.LSN.TEMP_LAYERS
        if self.n in self.temp_layers:
            self.Att_type = 'temporal'
        else:
            self.Att_type = 'standard'
        self.fr = cfg.DATA.NUM_INPUT_FRAMES

    def forward(self, x):
        if self.Att_type == 'temporal':
            x = x.view(int(x.size()[0] // self.fr), self.fr, *x.size()[1:]).permute(0, 2, 1, 3)
            bs = x.size()[0]
            x = x.reshape(-1, self.fr, x.size()[-1])
            attn = self.attn(self.norm1(x))
            x = x + attn
            x = x.reshape(bs, -1, self.fr, x.size()[-1]).permute(0, 2, 1, 3)
            x = x.reshape(x.size()[0]*self.fr, -1, x.size()[-1])
            x = x + self.mlp(self.norm2(x))
            return x
        else:
            attn = self.attn(self.norm1(x))
            x = x + attn
            x = x + self.mlp(self.norm2(x))
            return x

class Adapter(nn.Module):
    def __init__(self, dim, cfg):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=8, qkv_bias=True)
        self.res_alpha = cfg.DATA.RES_ALPHA

    def forward(self, x):
        attn = self.attn(self.norm1(x))
        x = (2.0 - self.res_alpha) * x + self.res_alpha * attn
        return x

class LSN(nn.Module):
    def __init__(self, clip_model, cfg):
        super().__init__()
        self.embeddings = copy.deepcopy(clip_model.vision_model.embeddings)
        self.pre_layernorm = copy.deepcopy(clip_model.vision_model.pre_layrnorm)
        self.layers = nn.ModuleList(copy.deepcopy(clip_model.vision_model.encoder.layers[:]))
        self.post_layernorm = copy.deepcopy(clip_model.vision_model.post_layernorm)
        self.projection = copy.deepcopy(clip_model.visual_projection)
        self.last_dim = clip_model.vision_model.config.projection_dim
        # cfg-related settings
        self.num_layers = cfg.LSN.NUM_LAYERS
        self.hidden_dim = cfg.LSN.HIDDEN_DIM
        self.output_dim = cfg.LSN.OUTPUT_DIM
        self.adapter_dim = cfg.LSN.ADAPTER_DIM
        self.use_gate = cfg.LSN.USE_GATE
        self.remove_clip_proj = cfg.LSN.REMOVE_PROJ
        self.fr = cfg.DATA.NUM_INPUT_FRAMES
        self.test = cfg.TEST.ENABLE
        # side downsample linear layers
        self.side_downsample = nn.ModuleList([
            nn.Linear(self.hidden_dim, int(self.output_dim), bias=False)
            for _ in range(self.num_layers + 1)
        ])
        # set embed dim for block cfg
        cfg.LSN.EMBED_DIM = int(self.output_dim)
        self.side_transformer = nn.ModuleList([
            ViTBlock(cfg, n)
            for n in range(self.num_layers)
        ])
        if not self.remove_clip_proj:
            self.side_LN = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers + 1)])
            self.side_last_LN = nn.LayerNorm(self.last_dim)
        else:
            self.side_LN = nn.ModuleList([nn.LayerNorm(self.hidden_dim) for _ in range(self.num_layers + 1)])
        self.Adapter = Adapter(self.adapter_dim,cfg)
        self.up_linear = nn.Linear(self.output_dim, self.adapter_dim, bias=False)
        self.up_ln = nn.LayerNorm(self.output_dim)
        if self.use_gate == "learnable":
            if not self.remove_clip_proj:
                self.side_gate_params = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * cfg.LSN.GATE_ALPHA) for _ in range(self.num_layers + 1)]
                )
            else:
                self.side_gate_params = nn.ParameterList(
                    [nn.Parameter(torch.ones(1) * cfg.LSN.GATE_ALPHA) for _ in range(self.num_layers)]
                )
            self.gate_T = cfg.LSN.GATE_T
        if not self.remove_clip_proj:
            self.side_proj_downsample = nn.Linear(self.last_dim, self.output_dim, bias=False)
            self.side_post_LN = nn.LayerNorm(int(self.output_dim))
            self.side_reduce = nn.Linear(int(self.output_dim), self.output_dim, bias=False)
        if not self.test:
            self.prune_and_initialize(cfg)

    def PCA(self, Wp, target_coldim, target_rowdim):
        """PCA-based conversion for weight / bias tensors."""
        if len(Wp.shape) == 1:
            W_reduce = F.interpolate(Wp.unsqueeze(0).unsqueeze(0).unsqueeze(-1),
                                     size=(target_rowdim, 1), mode='bicubic')
            W_init = W_reduce.squeeze(0).squeeze(0).squeeze(-1)
            return W_init
        else:
            W_np = Wp.numpy()
            if target_rowdim >= target_coldim:
                pca_col = PCA(n_components=target_coldim)
                W_pca = pca_col.fit_transform(W_np)
                W_init = W_pca[:target_rowdim, :]
            else:
                pca = PCA(n_components=target_rowdim)
                W_pca = pca.fit_transform(W_np.T).T
                W_init = W_pca[:, :target_coldim]
            return torch.tensor(W_init)
    def prune_and_initialize(self, cfg):
        """
        Initializes the LSN weights by pruning parameters from the pre-trained CLIP backbone.
        """
        with torch.no_grad():
            for i, side_layer in enumerate(self.side_transformer):
                pretrained_layer = self.layers[i]
                try:
                    q_weight = pretrained_layer.self_attn.q_proj.weight.data
                    k_weight = pretrained_layer.self_attn.k_proj.weight.data
                    v_weight = pretrained_layer.self_attn.v_proj.weight.data
                    out_proj_weight = pretrained_layer.self_attn.out_proj.weight.data

                    q_bias = pretrained_layer.self_attn.q_proj.bias.data
                    k_bias = pretrained_layer.self_attn.k_proj.bias.data
                    v_bias = pretrained_layer.self_attn.v_proj.bias.data
                    out_proj_bias = pretrained_layer.self_attn.out_proj.bias.data

                    q_weight_tune = self.PCA(q_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    k_weight_tune = self.PCA(k_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    v_weight_tune = self.PCA(v_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    out_proj_weight_tune = self.PCA(out_proj_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)

                    q_bias_tune = self.PCA(q_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    k_bias_tune = self.PCA(k_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    v_bias_tune = self.PCA(v_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    out_proj_bias_tune = self.PCA(out_proj_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)

                    side_layer.attn.q_linear.weight.data.copy_(copy.deepcopy(q_weight_tune))
                    side_layer.attn.k_linear.weight.data.copy_(copy.deepcopy(k_weight_tune))
                    side_layer.attn.v_linear.weight.data.copy_(copy.deepcopy(v_weight_tune))
                    side_layer.attn.proj.weight.data.copy_(copy.deepcopy(out_proj_weight_tune))

                    side_layer.attn.q_linear.bias.data.copy_(copy.deepcopy(q_bias_tune))
                    side_layer.attn.k_linear.bias.data.copy_(copy.deepcopy(k_bias_tune))
                    side_layer.attn.v_linear.bias.data.copy_(copy.deepcopy(v_bias_tune))
                    side_layer.attn.proj.bias.data.copy_(copy.deepcopy(out_proj_bias_tune))
                except Exception:
                    pass
                try:
                    layernorm1_weight = pretrained_layer.layer_norm1.weight.data
                    layernorm2_weight = pretrained_layer.layer_norm2.weight.data
                    layernorm1_bias = pretrained_layer.layer_norm1.bias.data
                    layernorm2_bias = pretrained_layer.layer_norm2.bias.data

                    layernorm1_weight_tune = self.PCA(layernorm1_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    layernorm2_weight_tune = self.PCA(layernorm2_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    layernorm1_bias_tune = self.PCA(layernorm1_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                    layernorm2_bias_tune = self.PCA(layernorm2_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)

                    side_layer.norm1.weight.data.copy_(copy.deepcopy(layernorm1_weight_tune))
                    side_layer.norm2.weight.data.copy_(copy.deepcopy(layernorm2_weight_tune))
                    side_layer.norm1.bias.data.copy_(copy.deepcopy(layernorm1_bias_tune))
                    side_layer.norm2.bias.data.copy_(copy.deepcopy(layernorm2_bias_tune))
                except Exception:
                    pass
                try:
                    fc1_weight = pretrained_layer.mlp.fc1.weight.data
                    fc2_weight = pretrained_layer.mlp.fc2.weight.data
                    fc1_bias = pretrained_layer.mlp.fc1.bias.data
                    fc2_bias = pretrained_layer.mlp.fc2.bias.data

                    fc1_weight_tune = self.PCA(fc1_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM * 4)
                    fc2_weight_tune = self.PCA(fc2_weight, cfg.LSN.EMBED_DIM * 4, cfg.LSN.EMBED_DIM)
                    fc1_bias_tune = self.PCA(fc1_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM * 4)
                    fc2_bias_tune = self.PCA(fc2_bias, cfg.LSN.EMBED_DIM * 4, cfg.LSN.EMBED_DIM)

                    side_layer.mlp.fc1.weight.data.copy_(copy.deepcopy(fc1_weight_tune))
                    side_layer.mlp.fc2.weight.data.copy_(copy.deepcopy(fc2_weight_tune))
                    side_layer.mlp.fc1.bias.data.copy_(copy.deepcopy(fc1_bias_tune))
                    side_layer.mlp.fc2.bias.data.copy_(copy.deepcopy(fc2_bias_tune))
                except Exception:
                    pass
                if i == self.num_layers-1:
                    q_weight_tune = self.PCA(q_weight, self.adapter_dim, self.adapter_dim)
                    k_weight_tune = self.PCA(k_weight, self.adapter_dim, self.adapter_dim)
                    v_weight_tune = self.PCA(v_weight, self.adapter_dim, self.adapter_dim)
                    out_proj_weight_tune = self.PCA(out_proj_weight, self.adapter_dim, self.adapter_dim)
                    
                    q_bias_tune = self.PCA(q_bias, self.adapter_dim, self.adapter_dim)
                    k_bias_tune = self.PCA(k_bias, self.adapter_dim, self.adapter_dim)
                    v_bias_tune = self.PCA(v_bias, self.adapter_dim, self.adapter_dim)
                    out_proj_bias_tune = self.PCA(out_proj_bias, self.adapter_dim, self.adapter_dim)
                
                    self.Adapter.attn.q_linear.weight.data.copy_(copy.deepcopy(q_weight_tune))
                    self.Adapter.attn.k_linear.weight.data.copy_(copy.deepcopy(k_weight_tune))
                    self.Adapter.attn.v_linear.weight.data.copy_(copy.deepcopy(v_weight_tune))
                    self.Adapter.attn.proj.weight.data.copy_(copy.deepcopy(out_proj_weight_tune))
                    
                    self.Adapter.attn.q_linear.bias.data.copy_(copy.deepcopy(q_bias_tune))
                    self.Adapter.attn.k_linear.bias.data.copy_(copy.deepcopy(k_bias_tune))
                    self.Adapter.attn.v_linear.bias.data.copy_(copy.deepcopy(v_bias_tune))
                    self.Adapter.attn.proj.bias.data.copy_(copy.deepcopy(out_proj_bias_tune))
                if i == self.num_layers-1 and (not self.remove_clip_proj):
                    try:
                        post_layernorm_weight = self.post_layernorm.weight.data
                        post_layernorm_bias = self.post_layernorm.bias.data

                        post_layernorm_weight_tune = self.PCA(post_layernorm_weight, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)
                        post_layernorm_bias_tune = self.PCA(post_layernorm_bias, cfg.LSN.EMBED_DIM, cfg.LSN.EMBED_DIM)

                        self.side_post_LN.weight.data.copy_(copy.deepcopy(post_layernorm_weight_tune))
                        self.side_post_LN.bias.data.copy_(copy.deepcopy(post_layernorm_bias_tune))

                        pro_weight = self.projection.weight.data
                        fc1_weight_tune = self.PCA(pro_weight, cfg.LSN.EMBED_DIM, cfg.LSN.OUTPUT_DIM)
                        self.side_reduce.weight.data.copy_(copy.deepcopy(fc1_weight_tune))
                    except Exception:
                        pass
                print(f"the {i+1} th side layer is initialized...")

    def forward(self, x):
        # x: tokens/patch embeddings input (as in CLIP vision pipeline)
        x_ = self.embeddings(x)
        hidden_clip = self.pre_layernorm(x_)
        hidden_clip_LN = self.side_LN[0](hidden_clip)
        hidden_downsample_0 = self.side_downsample[0](hidden_clip_LN)
        for i, clip_layer in enumerate(self.layers):
            side_LN = self.side_LN[i+1]
            side_layer = self.side_transformer[i]
            side_downsample = self.side_downsample[i+1]
            hidden_clip = clip_layer(hidden_clip, attention_mask=None, causal_attention_mask=None)[0]
            hidden_downsample = side_downsample(side_LN(hidden_clip))
            if i == 0:
                side_hidden = side_layer(hidden_downsample_0)
            else:
                side_hidden = side_layer(hidden_state)
            if self.use_gate == "learnable":
                side_gate = self.side_gate_params[i]
                gate = torch.sigmoid(side_gate / self.gate_T)
                hidden_state = gate * side_hidden + (1 - gate) * hidden_downsample
            else:
                hidden_state = side_hidden + hidden_downsample
        if not self.remove_clip_proj:
            hidden_clip = self.post_layernorm(hidden_clip)
            hidden_clip = self.projection(hidden_clip)
            hidden_downsample = self.side_proj_downsample(self.side_last_LN(hidden_clip))
            side_hidden = self.side_reduce(self.side_post_LN(hidden_state))
            if self.use_gate == "learnable":
                side_gate = self.side_gate_params[-1]
                gate = torch.sigmoid(side_gate / self.gate_T)
                hidden_state = gate * side_hidden + (1 - gate) * hidden_downsample
            else:
                hidden_state = side_hidden + hidden_downsample
        hidden_clip = self.post_layernorm(hidden_clip)
        hidden_clip = self.projection(hidden_clip)
        return hidden_state, hidden_clip[:, 0]  

def build_model(cfg):
    print("Construct the model: ", cfg.VIDEO.HEAD.BACKBONE_NAME)
    text_model = CLIPTextModelWithProjection.from_pretrained(
        pretrained_model_name_or_path=f"./model/{cfg.VIDEO.HEAD.BACKBONE_NAME}",
        local_files_only=True,
        ignore_mismatched_sizes=True,
        config=f"./model/{cfg.VIDEO.HEAD.BACKBONE_NAME}/config.json"
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        pretrained_model_name_or_path=f"./model/{cfg.VIDEO.HEAD.BACKBONE_NAME}/",
        local_files_only=True,
    )
    image_model = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_model_name_or_path=f"./model/{cfg.VIDEO.HEAD.BACKBONE_NAME}/",
        local_files_only=True,
        ignore_mismatched_sizes=True,
        config=f"./model/{cfg.VIDEO.HEAD.BACKBONE_NAME}/config.json"
    )
    print("A CLIP {} is loaded !".format(cfg.VIDEO.HEAD.BACKBONE_NAME))
    return image_model, text_model.cuda(), tokenizer

