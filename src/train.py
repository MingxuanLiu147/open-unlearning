"""
Unlearning 训练主入口
====================

本脚本是基于 Hydra 配置框架的 unlearning 训练入口。

核心流程：
1. 通过 Hydra 加载和合并配置文件（支持命令行覆盖）
2. 初始化随机种子，确保实验可复现
3. 加载预训练模型和 tokenizer
4. 加载训练数据（forget 集和 retain 集）
5. 初始化 unlearning trainer（如 SimNPO、GradAscent 等）
6. 执行训练和评估

使用示例：
    python src/train.py \\
        --config-name=unlearn.yaml \\
        experiment=unlearn/tofu/default \\
        trainer=SimNPO \\
        forget_split=forget10 \\
        retain_split=retain90
"""

import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """训练主函数
    
    通过 Hydra 装饰器自动加载配置文件，并支持命令行参数覆盖。
    
    Args:
        cfg (DictConfig): Hydra 配置对象，包含以下主要部分：
            - model: 模型配置（模型路径、注意力实现等）
            - data: 数据配置（forget/retain 数据集、split 等）
            - trainer: 训练器配置（算法类型、超参数等）
            - eval: 评估配置（评估指标、评估频率等）
            - mode: 训练模式（"train" 或 "unlearn"）
    
    配置加载顺序：
        1. 加载 config_name 指定的基础配置（如 unlearn.yaml）
        2. 应用 experiment 配置覆盖（如 experiment/unlearn/tofu/default.yaml）
        3. 应用命令行参数覆盖（如 trainer=SimNPO）
    """
    
    # ==================== 步骤1: 设置随机种子 ====================
    # 确保实验可复现（固定模型初始化、数据打乱、dropout 等随机性）
    seed_everything(cfg.trainer.args.seed)
    
    # ==================== 步骤2: 加载模型配置 ====================
    # mode 决定数据加载方式：
    #   - "train": 普通训练，按 split 返回数据
    #   - "unlearn": 遗忘训练，将 forget/retain 合并为 ForgetRetainDataset
    mode = cfg.get("mode", "train")
    
    model_cfg = cfg.model
    # template_args 包含对话模板相关配置（用于格式化输入输出）
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    
    # 加载预训练模型和 tokenizer
    # 返回值：
    #   - model: HuggingFace 预训练模型（如 LLaMA、Mistral 等）
    #   - tokenizer: 对应的 tokenizer
    model, tokenizer = get_model(model_cfg)

    # ==================== 步骤3: 加载数据集 ====================
    data_cfg = cfg.data
    # get_data 根据 mode 返回不同格式的数据：
    #   - mode="train": {"forget": Dataset, "retain": Dataset, "eval": Dataset}
    #   - mode="unlearn": {"train": ForgetRetainDataset, "eval": Dataset}
    #     其中 ForgetRetainDataset 每次返回 {"forget": sample, "retain": sample}
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # ==================== 步骤4: 加载数据整理器 (Collator) ====================
    # Collator 负责将多个样本整理成一个 batch
    # 主要功能：padding、截断、生成 attention_mask 等
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # ==================== 步骤5: 初始化 Trainer ====================
    trainer_cfg = cfg.trainer
    # trainer_cfg 包含：
    #   - handler: Trainer 类名（如 "SimNPO"、"GradAscent"）
    #   - args: TrainingArguments（学习率、batch size、训练轮数等）
    #   - method_args: 算法特定参数（如 SimNPO 的 delta、beta、gamma 等）
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # ==================== 步骤6: 加载评估器 ====================
    # 评估器用于在训练过程中或训练后评估模型性能
    # 常见评估指标：ROUGE、概率差异、模型效用等
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    if eval_cfgs:
        evaluators = get_evaluators(
            eval_cfgs=eval_cfgs,
            template_args=template_args,
            model=model,
            tokenizer=tokenizer,
        )

    # ==================== 步骤7: 实例化 Trainer ====================
    # load_trainer 根据 trainer_cfg.handler 从注册表中选择对应的 Trainer 类
    # 例如：handler="SimNPO" → 实例化 SimNPO 类
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),  # ForgetRetainDataset（unlearn模式）
        eval_dataset=data.get("eval", None),     # 评估数据集
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,                   # 评估器列表
        template_args=template_args,
    )

    # ==================== 步骤8: 执行训练 ====================
    if trainer_args.do_train:
        # 开始训练循环
        # 训练过程中会：
        #   1. 从 ForgetRetainDataset 采样 batch
        #   2. 调用 trainer.compute_loss() 计算损失
        #   3. 反向传播更新模型参数
        #   4. 定期评估和保存模型
        trainer.train()
        
        # 保存训练状态（optimizer、scheduler、随机数生成器状态等）
        trainer.save_state()
        
        # 保存最终模型权重
        trainer.save_model(trainer_args.output_dir)

    # ==================== 步骤9: 执行评估 ====================
    if trainer_args.do_eval:
        # 在评估集上评估模型性能
        # 返回评估指标（如 loss、ROUGE、准确率等）
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
