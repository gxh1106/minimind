from mmengine.config import read_base

# 假设你使用 C-Eval 数据集作为中文评估基准（这是中文 LLM 最常用的 PPL/准确率测试集之一）
# 你也可以在 opencompass/configs/datasets/ppl 目录下选择其他中文数据集
with read_base():
#     from opencompass.configs.datasets.ceval.ceval_ppl import ceval_datasets
    from opencompass.configs.datasets.cmmlu.cmmlu_ppl import cmmlu_datasets

# 定义模型配置
models = [
    dict(
        type='HuggingFaceCausalLM',
        path='/mnt/nvme0n1/users/gxh/llm/MiniMind2-GatedAttn-Pretain-512',
        tokenizer_path='/mnt/nvme0n1/users/gxh/llm/MiniMind2-GatedAttn-Pretain-512',
        # 这里设置 max_seq_len
        max_seq_len=340, 
        tokenizer_kwargs=dict(
            padding_side='left', 
            truncation='left', 
            trust_remote_code=True
        ),
        model_kwargs=dict(
            device_map='auto', 
            trust_remote_code=True
        ),
        batch_size=4,  # 如果显存允许，可以适当调大
    )
]

# 将模型应用于数据集
datasets = cmmlu_datasets

# 运行配置
infer = dict(
    partitioner=dict(type='NaivePartitioner'),
    runner=dict(
        type='LocalRunner',
        max_num_workers=1,
        task=dict(type='OpenICLInferTask'),
    ),
)

# opencompass eval_ppl.py