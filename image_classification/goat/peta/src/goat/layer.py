import torch
import torch.nn as nn
import torch.nn.functional as F

from peft.tuners.tuners_utils import BaseTunerLayer


def _pad_to_multiple(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def _make_swift_permutation(num_blocks: int, block_width: int) -> torch.Tensor:
    indices = torch.arange(num_blocks * block_width, dtype=torch.long)
    return indices.reshape(num_blocks, block_width).transpose(0, 1).reshape(-1).contiguous()


class GOATLayer(BaseTunerLayer):
    adapter_layer_names = ("lora_swift_blocks",)

    def __init__(self, base_layer: nn.Module, num_blocks: int = 64, block_rank: int = 0):
        super().__init__()
        self.base_layer = base_layer

        if not isinstance(base_layer, nn.Linear):
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.N = num_blocks
        self.r = block_rank

        self.padded_in_features = _pad_to_multiple(self.in_features, self.N)
        self.padded_out_features = _pad_to_multiple(self.out_features, self.N)
        self.d1 = self.padded_in_features // self.N
        self.d2 = self.padded_out_features // self.N
        self.register_buffer("lora_swift_perm", _make_swift_permutation(self.N, self.d2), persistent=True)
        self.lora_swift_blocks = nn.ParameterDict()

    def update_layer(self, adapter_name: str, **kwargs) -> None:
        self.lora_swift_blocks[adapter_name] = nn.Parameter(
            torch.zeros(self.N, self.d1, self.d2),
            requires_grad=True,
        )

        for param in self.base_layer.parameters():
            param.requires_grad = False
        self._move_adapter_to_device_of_base_layer(adapter_name)

    def reset_parameters(self, adapter_name: str, init_lora_weights: bool) -> None:
        if init_lora_weights and adapter_name in self.lora_swift_blocks:
            nn.init.zeros_(self.lora_swift_blocks[adapter_name])


class LinearGOATLayer(nn.Module, GOATLayer):
    def __init__(
        self,
        base_layer: nn.Module,
        adapter_name: str,
        num_blocks: int = 64,
        block_rank: int = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        GOATLayer.__init__(self, base_layer=base_layer, num_blocks=num_blocks, block_rank=block_rank)
        self._active_adapter = adapter_name
        self.update_layer(adapter_name=adapter_name)
        self.to(self.base_layer.weight.device)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)

        if getattr(self, "disable_adapters", False):
            return result.to(previous_dtype)

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_swift_blocks:
                continue

            swift_blocks = self.lora_swift_blocks[active_adapter]
            batch_dims = x.shape[:-1]
            xd = x.to(swift_blocks.dtype)
            if self.padded_in_features != self.in_features:
                xd = F.pad(xd, (0, self.padded_in_features - self.in_features))
            xd = xd.reshape(*batch_dims, self.N, self.d1)
            xd = torch.einsum("...nd,ndo->...no", xd, swift_blocks)
            xd = xd.reshape(*batch_dims, self.padded_out_features)
            xd = xd.index_select(-1, self.lora_swift_perm)
            xd = xd[..., : self.out_features]
            result = result + xd.to(result.dtype)

        return result.to(previous_dtype)
