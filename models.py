# models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Iterable, Tuple

import torch
import torch.nn as nn

Tensor = torch.Tensor
State = Dict[str, Tensor]


# =============================================================================
# helpers
# =============================================================================

def _conv7x7(in_planes: int, out_planes: int, stride: int = 2) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)


def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def _auto_stem_type(input_size: int) -> str:
    """
    32x32 -> CIFAR-style stem
    >=128 -> ImageNet/HAM-style stem
    """
    return "imagenet" if int(input_size) >= 128 else "cifar"


# =============================================================================
# ResNet blocks
# =============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        width = planes
        self.conv1 = _conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = _conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = _conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNetBackbone(nn.Module):
    """
    ResNet backbone with configurable stem.

    stem_type="cifar":
        conv3x3, stride=1, no maxpool
    stem_type="imagenet":
        conv7x7, stride=2, maxpool
    """

    def __init__(
        self,
        block: type[nn.Module],
        layers: List[int],
        num_classes: int = 10,
        stem_type: str = "cifar",
    ):
        super().__init__()
        self.inplanes = 64
        self.stem_type = stem_type

        if stem_type == "cifar":
            self.conv1 = _conv3x3(3, 64, stride=1)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = None
        elif stem_type == "imagenet":
            self.conv1 = _conv7x7(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            raise ValueError(f"Unsupported stem_type: {stem_type}")

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make_layer(self, block: type[nn.Module], planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers_: List[nn.Module] = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_.append(block(self.inplanes, planes, 1, None))
        return nn.Sequential(*layers_)

    def forward(self, x: Tensor) -> Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def _make_resnet(arch: str, num_classes: int, stem_type: str) -> ResNetBackbone:
    arch = arch.lower()
    if arch == "resnet18":
        return ResNetBackbone(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem_type=stem_type)
    if arch == "resnet34":
        return ResNetBackbone(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, stem_type=stem_type)
    if arch == "resnet50":
        return ResNetBackbone(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem_type=stem_type)
    raise ValueError(f"Unsupported ResNet arch: {arch}")


class ResNetForAgg(nn.Module):
    """
    Full model with cut API.
    cut_idx counts residual blocks only; stem is always included in prefix.
    """

    def __init__(
        self,
        arch: str = "resnet34",
        num_classes: int = 10,
        input_size: int = 32,
        stem_type: Optional[str] = None,
    ):
        super().__init__()
        if stem_type is None:
            stem_type = _auto_stem_type(input_size)

        self.backbone = _make_resnet(arch, num_classes, stem_type=stem_type)
        self.arch = arch
        self.num_classes = num_classes
        self.input_size = int(input_size)
        self.stem_type = stem_type

        blocks: List[nn.Module] = []
        for layer in [self.backbone.layer1, self.backbone.layer2, self.backbone.layer3, self.backbone.layer4]:
            for b in layer:
                blocks.append(b)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def num_blocks(self) -> int:
        return len(self.blocks)

    def client_module_prefixes_by_cut(self, cut_idx: int) -> List[str]:
        """
        Module prefixes for prefix (client-side) region.
        Stem is always on client side for ResNet.
        """
        B = self.num_blocks()
        if not (0 <= cut_idx <= B):
            raise ValueError(f"cut_idx out of range [0,{B}]")

        prefixes: List[str] = ["backbone.conv1", "backbone.bn1"]
        # maxpool has no parameters/buffers, so whether included or not does not affect key expansion

        stage_sizes = [
            len(self.backbone.layer1),
            len(self.backbone.layer2),
            len(self.backbone.layer3),
            len(self.backbone.layer4),
        ]
        remaining = cut_idx
        stage = 1
        for sz in stage_sizes:
            take = min(remaining, sz)
            for j in range(take):
                prefixes.append(f"backbone.layer{stage}.{j}")
            remaining -= take
            stage += 1
            if remaining <= 0:
                break
        return prefixes

    def server_module_prefixes_from_block(self, start_block: int) -> List[str]:
        """
        Module prefixes for suffix (server-side) region (includes head).
        """
        B = self.num_blocks()
        if not (0 <= start_block <= B):
            raise ValueError(f"start_block out of range [0,{B}]")

        if start_block == B:
            return ["backbone.avgpool", "backbone.fc"]

        stage_sizes = [
            len(self.backbone.layer1),
            len(self.backbone.layer2),
            len(self.backbone.layer3),
            len(self.backbone.layer4),
        ]
        idx = start_block
        stage = 1
        for sz in stage_sizes:
            if idx >= sz:
                idx -= sz
                stage += 1
            else:
                break

        if stage > 4:
            return ["backbone.avgpool", "backbone.fc"]

        prefixes: List[str] = []
        for j in range(idx, stage_sizes[stage - 1]):
            prefixes.append(f"backbone.layer{stage}.{j}")
        for st in range(stage + 1, 5):
            for j in range(stage_sizes[st - 1]):
                prefixes.append(f"backbone.layer{st}.{j}")

        prefixes.append("backbone.avgpool")
        prefixes.append("backbone.fc")
        return prefixes
    
    def build_profile_modules(self, cut_idx: int):
        """
        Return (client_module, server_module) for profiling.
        client_module: input -> smashed data
        server_module: smashed data -> logits
        """
        B = self.num_blocks()
        if not (0 <= cut_idx <= B):
            raise ValueError(f"cut_idx out of range [0,{B}]")

        bb = self.backbone

        client_layers = [bb.conv1, bb.bn1, bb.relu]
        if bb.maxpool is not None:
            client_layers.append(bb.maxpool)
        for i in range(cut_idx):
            client_layers.append(self.blocks[i])

        server_layers = []
        for i in range(cut_idx, B):
            server_layers.append(self.blocks[i])
        server_layers.extend([bb.avgpool, nn.Flatten(1), bb.fc])

        return nn.Sequential(*client_layers), nn.Sequential(*server_layers)

# =============================================================================
# VGG backbone with the same cut API
# =============================================================================

_VGG_STAGE_CFGS = {
    "vgg11": [1, 1, 2, 2, 2],
    "vgg13": [2, 2, 2, 2, 2],
    "vgg16": [2, 2, 3, 3, 3],
    "vgg19": [2, 2, 4, 4, 4],
}


class VGGBackbone(nn.Module):
    """
    VGG-style backbone with stage blocks.
    Each stage = repeated conv(+bn)+relu + final maxpool.

    Notes:
      - This is kept intentionally lightweight:
        global avgpool + linear head
      - It is not the exact torchvision VGG classifier head
      - The benefit is that it aligns well with your split/bank code
    """

    def __init__(
        self,
        arch: str = "vgg16",
        num_classes: int = 10,
        batch_norm: bool = True,
    ):
        super().__init__()
        arch = arch.lower()
        if arch not in _VGG_STAGE_CFGS:
            raise ValueError(f"Unsupported VGG arch: {arch}")

        stage_convs = _VGG_STAGE_CFGS[arch]
        stage_channels = [64, 128, 256, 512, 512]

        self.arch = arch
        self.batch_norm = batch_norm

        stages: List[nn.Module] = []
        in_ch = 3
        for n_conv, out_ch in zip(stage_convs, stage_channels):
            layers: List[nn.Module] = []
            for _ in range(n_conv):
                layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False))
                if batch_norm:
                    layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
                in_ch = out_ch
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            stages.append(nn.Sequential(*layers))

        self.stages = nn.ModuleList(stages)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: Tensor) -> Tensor:
        for st in self.stages:
            x = st(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class VGGForAgg(nn.Module):
    """
    Full VGG-style model with cut API.
    cut_idx counts stage blocks.
    cut_idx=0 means prefix is empty.
    """

    def __init__(
        self,
        arch: str = "vgg16",
        num_classes: int = 10,
        input_size: int = 224,
        batch_norm: bool = True,
    ):
        super().__init__()
        self.backbone = VGGBackbone(arch=arch, num_classes=num_classes, batch_norm=batch_norm)
        self.arch = arch
        self.num_classes = num_classes
        self.input_size = int(input_size)
        self.batch_norm = batch_norm

        self.blocks = self.backbone.stages

    def forward(self, x: Tensor) -> Tensor:
        return self.backbone(x)

    def num_blocks(self) -> int:
        return len(self.blocks)

    def client_module_prefixes_by_cut(self, cut_idx: int) -> List[str]:
        """
        Prefix uses VGG stages [0, cut_idx).
        """
        B = self.num_blocks()
        if not (0 <= cut_idx <= B):
            raise ValueError(f"cut_idx out of range [0,{B}]")

        prefixes: List[str] = []
        for i in range(cut_idx):
            prefixes.append(f"backbone.stages.{i}")
        return prefixes

    def server_module_prefixes_from_block(self, start_block: int) -> List[str]:
        """
        Suffix uses VGG stages [start_block, B) + head.
        """
        B = self.num_blocks()
        if not (0 <= start_block <= B):
            raise ValueError(f"start_block out of range [0,{B}]")

        prefixes: List[str] = []
        for i in range(start_block, B):
            prefixes.append(f"backbone.stages.{i}")
        prefixes.append("backbone.avgpool")
        prefixes.append("backbone.classifier")
        return prefixes
    
    def build_profile_modules(self, cut_idx: int):
        """
        Return (client_module, server_module) for profiling.
        """
        B = self.num_blocks()
        if not (0 <= cut_idx <= B):
            raise ValueError(f"cut_idx out of range [0,{B}]")

        bb = self.backbone

        client_layers = []
        for i in range(cut_idx):
            client_layers.append(bb.stages[i])

        server_layers = []
        for i in range(cut_idx, B):
            server_layers.append(bb.stages[i])
        server_layers.extend([bb.avgpool, nn.Flatten(1), bb.classifier])

        return nn.Sequential(*client_layers), nn.Sequential(*server_layers)


# =============================================================================
# unified builder
# =============================================================================

def build_model(
    arch: str = "resnet34",
    num_classes: int = 10,
    input_size: int = 32,
    stem_type: Optional[str] = None,
    vgg_batch_norm: bool = True,
) -> nn.Module:
    """
    Unified entry for future profiler/engine use.

    Examples:
      build_model("resnet18", num_classes=10, input_size=32)
      build_model("resnet18", num_classes=7, input_size=224)
      build_model("vgg16",    num_classes=7, input_size=224)
    """
    arch_l = arch.lower()
    if arch_l.startswith("resnet"):
        return ResNetForAgg(
            arch=arch_l,
            num_classes=num_classes,
            input_size=input_size,
            stem_type=stem_type,
        )
    if arch_l.startswith("vgg"):
        return VGGForAgg(
            arch=arch_l,
            num_classes=num_classes,
            input_size=input_size,
            batch_norm=vgg_batch_norm,
        )
    raise ValueError(f"Unsupported architecture: {arch}")


# =============================================================================
# Bank
# =============================================================================

@dataclass(frozen=True)
class Keysets:
    cut_idx: int
    client_float_keys: List[str]
    server_float_keys: List[str]


class ClientModelBank:
    """
    One live model + N+1 snapshot states on device.

    Authority of:
      - concrete keys = named_parameters + named_buffers
      - float/non-float partition
      - cut keysets expansion (prefix->concrete keys) + caching
      - primitive in-place ops:
          1) snapshot -> model (all keys)
          2) model -> snapshot (all keys)
          3) weighted average over snapshots into snapshot (float keys)
          4) delta apply: snapshot += a*(model-base) (float keys)
    """

    def __init__(
        self,
        num_clients: int,
        model: nn.Module,
        device: torch.device,
        preallocate_tmp: bool = True,
    ):
        if num_clients <= 0:
            raise ValueError("num_clients must be positive")

        self.num_clients = int(num_clients)
        self.device = device

        self.model = model.to(device)
        self.model.train()

        # tensor refs (authority keys)
        self.tensor_ref: Dict[str, Tensor] = {}
        self.param_keys: List[str] = []
        self.buffer_keys: List[str] = []

        for n, p in self.model.named_parameters():
            self.tensor_ref[n] = p
            self.param_keys.append(n)
        for n, b in self.model.named_buffers():
            self.tensor_ref[n] = b
            self.buffer_keys.append(n)

        self.all_keys: List[str] = list(self.tensor_ref.keys())
        self.all_keys_set: Set[str] = set(self.all_keys)

        self.param_float_keys: List[str] = [k for k in self.param_keys if torch.is_floating_point(self.tensor_ref[k])]
        self.buffer_float_keys: List[str] = [k for k in self.buffer_keys if torch.is_floating_point(self.tensor_ref[k])]
        self.buffer_nonfloat_keys: List[str] = [k for k in self.buffer_keys if not torch.is_floating_point(self.tensor_ref[k])]

        # keep old interface names
        self.float_keys: List[str] = self.param_float_keys + self.buffer_float_keys
        self.nonfloat_keys: List[str] = [k for k in self.all_keys if not torch.is_floating_point(self.tensor_ref[k])]

        # cut support
        self._supports_cut = all(hasattr(self.model, a) for a in [
            "num_blocks",
            "client_module_prefixes_by_cut",
            "server_module_prefixes_from_block",
        ])
        self.B: Optional[int] = int(self.model.num_blocks()) if self._supports_cut else None
        self._keysets_cache: Dict[int, Keysets] = {}

        # prefix -> keys index (preprocess once)
        self._prefix2keys: Dict[str, List[str]] = self._build_prefix_index(self.all_keys)

        # snapshots
       
        self.server_state: State = self._alloc_state_like_model()
        self.client_base_state: Dict[int, State] = {}
        for cid in range(self.num_clients):
            st = self._alloc_state_like_model()
            self.copy_state_to_state(st, self.server_state)
            self.client_base_state[cid] = st

        # scratch
        self.tmp_base: Optional[State] = None
        self.tmp_delta: Optional[State] = None
        if preallocate_tmp:
            self.tmp_base = self._alloc_state_like_model()
            self.tmp_delta = self._alloc_state_like_model()

        self.client_prefix_anchor_state: Optional[List[State]] = None
        self._prefix_anchor_keyset_sig: Optional[Tuple[int, int]] = None
        self.client_full_anchor_state: Optional[List[State]] = None

    # -------------------------------------------------------------------------
    # alloc/copy
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _alloc_state_like_model(self) -> State:
        out: State = {}
        for k in self.all_keys:
            out[k] = self.tensor_ref[k].detach().clone()
        return out

    @torch.no_grad()
    def copy_state_to_model(self, src: State) -> None:
        for k in self.all_keys:
            self.tensor_ref[k].copy_(src[k])

    @torch.no_grad()
    def copy_model_to_state(self, dst: State) -> None:
        for k in self.all_keys:
            dst[k].copy_(self.tensor_ref[k])

    @torch.no_grad()
    def copy_state_to_state(self, dst: State, src: State, is_float_keys: bool = False) -> None:
        if is_float_keys:
            for k in self.float_keys:
                dst[k].copy_(src[k])
        else:
            for k in dst.keys():
                dst[k].copy_(src[k])

    @torch.no_grad()
    def update_buffers_into_state(self, dst: State, src: State, alpha: float = 1.0) -> None:
        """
        Simple buffer handling:
          - float buffers: EMA / averaging
          - non-float buffers: direct copy
        """
        a = float(alpha)
        for k in self.buffer_float_keys:
            dst[k].mul_(1.0 - a).add_(src[k] * a)
        for k in self.buffer_nonfloat_keys:
            dst[k].copy_(src[k])

    # -------------------------------------------------------------------------
    # keysets (cut)
    # -------------------------------------------------------------------------

    @staticmethod
    def _build_prefix_index(keys: List[str]) -> Dict[str, List[str]]:
        pref: Dict[str, List[str]] = {}
        for k in keys:
            parts = k.split(".")
            cur = []
            for i in range(len(parts)):
                cur.append(parts[i])
                p = ".".join(cur)
                pref.setdefault(p, []).append(k)
        return pref

    def _expand_prefixes(self, prefixes: Iterable[str]) -> Set[str]:
        out: Set[str] = set()
        for p in prefixes:
            out.update(self._prefix2keys.get(p, []))
        return out

    def get_keysets(self, cut_idx: int) -> Keysets:
        if not self._supports_cut:
            raise RuntimeError("Model does not support cut keysets")

        if self.B is None:
            raise RuntimeError("Internal error: B is None but supports_cut=True")

        cut_idx = int(cut_idx)
        if not (0 <= cut_idx <= self.B):
            raise ValueError(f"cut_idx out of range [0,{self.B}]")

        if cut_idx in self._keysets_cache:
            return self._keysets_cache[cut_idx]

        client_mods = self.model.client_module_prefixes_by_cut(cut_idx)
        server_mods = self.model.server_module_prefixes_from_block(cut_idx)

        client_keys = self._expand_prefixes(client_mods)
        server_keys = self._expand_prefixes(server_mods)

        miss = (client_keys | server_keys) - self.all_keys_set
        if miss:
            raise RuntimeError(f"Expanded keyset contains unknown keys, e.g. {list(miss)[:5]}")

        client_float = sorted([k for k in client_keys if torch.is_floating_point(self.tensor_ref[k])])
        server_float = sorted([k for k in server_keys if torch.is_floating_point(self.tensor_ref[k])])

        ks = Keysets(cut_idx=cut_idx, client_float_keys=client_float, server_float_keys=server_float)
        self._keysets_cache[cut_idx] = ks
        return ks

    # -------------------------------------------------------------------------
    # numeric primitives (float keys only)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def zero_float_keys_in_state(self, st: State, float_keys: List[str]) -> None:
        for k in float_keys:
            st[k].zero_()

    @torch.no_grad()
    def accumulate_weighted_state(self, dst: State, src: State, float_keys: List[str], weight: float) -> None:
        w = float(weight)
        for k in float_keys:
            dst[k].add_(w * src[k])

    @torch.no_grad()
    def assign_weighted_average_into_state(
        self,
        dst: State,
        src_states: List[State],
        weights: List[float],
        float_keys: List[str],
        normalize: bool = True,
    ) -> None:
        if len(src_states) != len(weights):
            raise ValueError("src_states and weights must have same length")
        if not src_states:
            raise ValueError("src_states cannot be empty")

        ws = [float(w) for w in weights]
        if normalize:
            s = sum(ws)
            if s <= 0:
                raise ValueError("sum(weights) must be positive when normalize=True")
            ws = [w / s for w in ws]

        self.zero_float_keys_in_state(dst, float_keys)
        for st, w in zip(src_states, ws):
            self.accumulate_weighted_state(dst, st, float_keys, w)

    @torch.no_grad()
    def apply_delta_from_model_into_state(self, dst: State, base: State, float_keys: List[str], scale: float) -> None:
        a = float(scale)
        for k in float_keys:
            dst[k].add_(a * (self.tensor_ref[k] - base[k]))

    # -------------------------------------------------------------------------
    # anchors
    # -------------------------------------------------------------------------

    def ensure_prefix_anchor(self, keysets: "Keysets") -> None:
        sig = (int(getattr(keysets, "cut_idx", -1)), int(len(keysets.client_float_keys)))
        if self.client_prefix_anchor_state is not None and self._prefix_anchor_keyset_sig == sig:
            return
        self._alloc_prefix_anchor(keysets)
        self._prefix_anchor_keyset_sig = sig

    def _alloc_prefix_anchor(self, keysets: "Keysets") -> None:
        num_clients = int(self.num_clients)
        ks = keysets

        anchors: List[State] = []
        with torch.no_grad():
            for _ in range(num_clients):
                st = self._alloc_state_like_model()
                for kk in ks.client_float_keys:
                    st[kk].copy_(self.server_state[kk])
                anchors.append(st)

        self.client_prefix_anchor_state = anchors

    @torch.no_grad()
    def copy_server_prefix_into_anchor(self, cid: int, keysets: "Keysets") -> None:
        if self.client_prefix_anchor_state is None:
            self._alloc_prefix_anchor(keysets)

        assert self.client_prefix_anchor_state is not None
        st = self.client_prefix_anchor_state[int(cid)]
        for kk in keysets.client_float_keys:
            st[kk].copy_(self.server_state[kk])

    def ensure_full_anchor(self) -> None:
        if self.client_full_anchor_state is not None:
            return

        anchors: List[State] = []
        with torch.no_grad():
            for _ in range(self.num_clients):
                st = self._alloc_state_like_model()
                for kk in self.float_keys:
                    st[kk].copy_(self.server_state[kk])
                anchors.append(st)
        self.client_full_anchor_state = anchors

    def copy_server_float_into_full_anchor(self, cid: int) -> None:
        if self.client_full_anchor_state is None:
            self.ensure_full_anchor()
        assert self.client_full_anchor_state is not None
        st = self.client_full_anchor_state[int(cid)]
        for kk in self.float_keys:
            st[kk].copy_(self.server_state[kk])


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("==== Test 1: ResNet34 / CIFAR-10 / 32x32 ====")
    m1 = build_model("resnet34", num_classes=10, input_size=32).to(device)
    x1 = torch.randn(4, 3, 32, 32, device=device)
    y1 = m1(x1)
    print("Output shape:", y1.shape)
    assert y1.shape == (4, 10)

    if hasattr(m1, "num_blocks"):
        B1 = m1.num_blocks()
        print("num_blocks =", B1)
        assert B1 == 16

        ks0 = m1.client_module_prefixes_by_cut(0)
        ksB = m1.server_module_prefixes_from_block(B1)
        print("client cut=0:", ks0)
        print("server cut=B:", ksB)

    bank1 = ClientModelBank(num_clients=3, model=m1, device=device)
    ks_bank_1 = bank1.get_keysets(4)
    print("Bank keysets (cut=4): prefix =", len(ks_bank_1.client_float_keys),
          ", suffix =", len(ks_bank_1.server_float_keys))
    assert len(ks_bank_1.client_float_keys) > 0
    assert len(ks_bank_1.server_float_keys) > 0

    print("\n==== Test 2: ResNet18 / HAM10000 / 224x224 ====")
    m2 = build_model("resnet18", num_classes=7, input_size=224).to(device)
    x2 = torch.randn(2, 3, 224, 224, device=device)
    y2 = m2(x2)
    print("Output shape:", y2.shape)
    assert y2.shape == (2, 7)

    if hasattr(m2, "stem_type"):
        print("stem_type =", m2.stem_type)
        assert m2.stem_type == "imagenet"

    bank2 = ClientModelBank(num_clients=2, model=m2, device=device)
    ks_bank_2 = bank2.get_keysets(2)
    print("Bank keysets (cut=2): prefix =", len(ks_bank_2.client_float_keys),
          ", suffix =", len(ks_bank_2.server_float_keys))
    assert len(ks_bank_2.client_float_keys) > 0
    assert len(ks_bank_2.server_float_keys) > 0

    print("\n==== Test 3: VGG16 / 224x224 ====")
    m3 = build_model("vgg16", num_classes=7, input_size=224, vgg_batch_norm=True).to(device)
    x3 = torch.randn(2, 3, 224, 224, device=device)
    y3 = m3(x3)
    print("Output shape:", y3.shape)
    assert y3.shape == (2, 7)

    if hasattr(m3, "num_blocks"):
        B3 = m3.num_blocks()
        print("num_blocks =", B3)
        assert B3 == 5

        cmods = m3.client_module_prefixes_by_cut(2)
        smods = m3.server_module_prefixes_from_block(2)
        print("client cut=2:", cmods)
        print("server from=2:", smods)

    bank3 = ClientModelBank(num_clients=2, model=m3, device=device)
    ks_bank_3 = bank3.get_keysets(2)
    print("Bank keysets (cut=2): prefix =", len(ks_bank_3.client_float_keys),
          ", suffix =", len(ks_bank_3.server_float_keys))
    assert len(ks_bank_3.client_float_keys) > 0
    assert len(ks_bank_3.server_float_keys) > 0

    print("\n==== Test 4: buffer update sanity ====")
    st_a = bank1._alloc_state_like_model()
    st_b = bank1._alloc_state_like_model()
    bank1.update_buffers_into_state(st_a, st_b, alpha=0.5)
    print("Buffer update passed.")

    print("\nAll model/bank tests passed.")