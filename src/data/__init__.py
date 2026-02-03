"""
Know-Surgery 数据集注册表
========================

统一管理所有数据集类，支持：
- Unlearning 数据集: QADataset, ForgetRetainDataset 等
- Editing 数据集: EditingDataset, ZSREDataset 等
- Injection 数据集: InjectDataset, AlpacaDataset 等
"""

from typing import Dict, Any, Union
from omegaconf import DictConfig

from data.qa import QADataset, QAwithIdkDataset, QAwithAlternateDataset
from data.collators import (
    DataCollatorForSupervisedDataset,
)
from data.unlearn import ForgetRetainDataset
from data.pretraining import PretrainingDataset, CompletionDataset

# Knowledge Editing 数据集
from data.editing import EditingDataset, ZSREDataset, CounterFactDataset

# Knowledge Injection 数据集
from data.inject import InjectDataset, AlpacaDataset, ShareGPTDataset

# 统一管理数据集类的注册表：
#   key: 类名字符串（如 "QADataset"）
#   value: 实际的 Dataset 类
DATASET_REGISTRY: Dict[str, Any] = {}
# 统一管理 collator 类的注册表
COLLATOR_REGISTRY: Dict[str, Any] = {}


def _register_data(data_class):
    """将数据集类注册到 DATASET_REGISTRY 中，便于通过字符串名称反射构造。"""
    DATASET_REGISTRY[data_class.__name__] = data_class


def _register_collator(collator_class):
    """将 collator 类注册到 COLLATOR_REGISTRY 中。"""
    COLLATOR_REGISTRY[collator_class.__name__] = collator_class


def _load_single_dataset(dataset_name, dataset_cfg: DictConfig, **kwargs):
    dataset_handler_name = dataset_cfg.get("handler")
    assert dataset_handler_name is not None, ValueError(
        f"{dataset_name} handler not set"
    )
    # 根据 handler 名称从注册表中取出真正的数据集类
    dataset_handler = DATASET_REGISTRY.get(dataset_handler_name)
    if dataset_handler is None:
        raise NotImplementedError(
            f"{dataset_handler_name} not implemented or not registered"
        )
    dataset_args = dataset_cfg.args
    # dataset_args 来自 yaml 配置，kwargs 一般来自上层（如 tokenizer、template_args）
    return dataset_handler(**dataset_args, **kwargs)


def get_datasets(dataset_cfgs: Union[Dict, DictConfig], **kwargs):
    """根据一组配置构造一个或多个数据集。

    - 如果只配置了一个数据集，则直接返回该 Dataset 实例
    - 如果配置了多个数据集，则返回一个 name -> Dataset 的字典
    """
    dataset = {}
    for dataset_name, dataset_cfg in dataset_cfgs.items():
        # access_name 允许通过 access_key 重命名（例如多个子数据集合并时使用）
        access_name = dataset_cfg.get("access_key", dataset_name)
        dataset[access_name] = _load_single_dataset(dataset_name, dataset_cfg, **kwargs)
    if len(dataset) == 1:
        # return a single dataset
        return list(dataset.values())[0]
    # return mapping to multiple datasets
    return dataset


def get_data(data_cfg: DictConfig, mode="train", **kwargs):
    """高层数据加载入口。

    根据 Hydra 的 data 配置构造完整的数据集字典，支持：
      - mode="train": 返回各个 split 原始数据集（如 {"forget": ds1, "retain": ds2, "eval": ds3}）
      - mode="unlearn": 将 forget / retain 等训练相关 split 组合成一个 ForgetRetainDataset，挂到 "train" 键下
    """
    data = {}
    # DictConfig -> 普通 dict，便于 pop / 迭代
    data_cfg = dict(data_cfg)
    # anchor 控制 ForgetRetainDataset 的锚定数据集（默认 forget）
    anchor = data_cfg.pop("anchor", "forget")
    for split, dataset_cfgs in data_cfg.items():
        # 对每个 split（forget/retain/eval/...）调用 get_datasets
        data[split] = get_datasets(dataset_cfgs, **kwargs)
    if mode == "train":
        # 普通训练：直接按 split 返回
        return data
    elif mode == "unlearn":
        # 遗忘场景：把除了 eval / test 以外的 split 合并成一个组合数据集
        unlearn_splits = {k: v for k, v in data.items() if k not in ("eval", "test")}
        # 典型情况：{"forget": QADataset, "retain": QADataset}
        unlearn_dataset = ForgetRetainDataset(**unlearn_splits, anchor=anchor)
        # 统一挂在 "train" 键下，便于 Trainer 使用
        data["train"] = unlearn_dataset
        # 清理掉原始的 forget / retain 键，只保留 train + eval/test
        for split in unlearn_splits:
            data.pop(split)
    return data


def _get_single_collator(collator_name: str, collator_cfg: DictConfig, **kwargs):
    # 从配置中拿到 collator 对应的类名
    collator_handler_name = collator_cfg.get("handler")
    assert collator_handler_name is not None, ValueError(
        f"{collator_name} handler not set"
    )
    # 从注册表中取出真正的 collator 类
    collator_handler = COLLATOR_REGISTRY.get(collator_handler_name)
    if collator_handler is None:
        raise NotImplementedError(
            f"{collator_handler_name} not implemented or not registered"
        )
    collator_args = collator_cfg.args
    # 构造 collator 实例（通常会传入 tokenizer、padding_side 等）
    return collator_handler(**collator_args, **kwargs)


def get_collators(collator_cfgs, **kwargs):
    """根据配置构造一个或多个 collator。

    - 只配置一个时，直接返回单个 collator
    - 多个时，返回 name -> collator 的字典
    """
    collators = {}
    for collator_name, collator_cfg in collator_cfgs.items():
        collators[collator_name] = _get_single_collator(
            collator_name, collator_cfg, **kwargs
        )
    if len(collators) == 1:
        # return a single collator
        return list(collators.values())[0]
    # return collators in a dict
    return collators


# Register datasets
_register_data(QADataset)
_register_data(QAwithIdkDataset)
_register_data(PretrainingDataset)
_register_data(CompletionDataset)
_register_data(QAwithAlternateDataset)

# Register composite datasets used in unlearning
# groups: unlearn
_register_data(ForgetRetainDataset)

# Register Knowledge Editing datasets
_register_data(EditingDataset)
_register_data(ZSREDataset)
_register_data(CounterFactDataset)

# Register Knowledge Injection datasets
_register_data(InjectDataset)
_register_data(AlpacaDataset)
_register_data(ShareGPTDataset)

# Register collators
_register_collator(DataCollatorForSupervisedDataset)
