from .dpo_dataset import make_dpo_data_module
from .grpo_dataset import make_grpo_data_module
from .sft_dataset import make_supervised_data_module
from .lvr_sft_dataset import make_supervised_data_module_lvr
from .lvr_sft_dataset_packed import make_packed_supervised_data_module_lvr
# NOTE: lvr_sft_dataset_packed_fixedToken.py is not present on this branch. Guard the import so the
# package still loads (the harness imports src.dataset for the SFT dataset class). Anything that
# actually needs the fixed-token packer gets None and fails only when it tries to use it.
try:
    from .lvr_sft_dataset_packed_fixedToken import make_packed_supervised_data_module_lvr_fixedToken
except ModuleNotFoundError:
    make_packed_supervised_data_module_lvr_fixedToken = None

__all__ =[
    "make_dpo_data_module",
    "make_supervised_data_module",
    "make_grpo_data_module",
    "make_supervised_data_module_lvr",
    "make_packed_supervised_data_module_lvr",
    "make_packed_supervised_data_module_lvr_fixedToken"
]