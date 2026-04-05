from huggingface_hub import HfApi, login

def push_model(repo_id, folder_path, path_in_repo, token=None):
    # 如果你不想每次都在代码里写 Token，可以去掉这一行，
    # 因为你之前已经执行过 `hf auth login` 了。
    # if token:
    #     login(token=token)
        
    api = HfApi()
    
    # 1. 创建仓库（仓库名已按你的要求修改）
    print(f"正在准备仓库: {repo_id}...")
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)
    
    # 2. 上传模型文件夹
    print(f"开始上传 {folder_path} 到 {repo_id} ...")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        path_in_repo=path_in_repo, # 这里指定仓库内的路径
        commit_message=f"Upload {path_in_repo} version"
    )
    print("上传成功！")

if __name__ == '__main__':
    # 填入你刚才获取的 Token (可选，如果已登录则不需要)
    # MY_TOKEN = "hf_XX"
    
    # 替换为你要求的仓库名和本地路径
    push_model(
        repo_id="XinghaoGuo/MiniMind3-GatedAttn", 
        folder_path="../MiniMind3-GatedAttn-PPO-768",
        path_in_repo="ppo-768" # 这里指定仓库内的路径,
        # token=MY_TOKEN
    )

# eval cli
# HF_DATASETS_TRUST_REMOTE_CODE=1 accelerate launch -m lm_eval \
#     --model hf \
#     --model_args pretrained=XinghaoGuo/MiniMind3-GatedAttn,subfolder=pretrain-512,trust_remote_code=True \
#     --tasks ceval-valid,cmmlu,aclue,tmmluplus \
#     --batch_size 64 \
#     --output_path ../results/minimind_eval_all.json
# lm_eval \
#     --model hf \
#     --model_args pretrained=XinghaoGuo/MiniMind3-GatedAttn,subfolder=pretrain-512,trust_remote_code=True,dtype=bfloat16 \
# pretrained=../MiniMind3-GatedAttn-Pretrain-512,trust_remote_code=True,dtype=auto
#     --tasks ceval-valid,cmmlu,aclue,tmmluplus \
#     --batch_size 64 \
#     --output_path ../results/minimind_eval_ceval.json

# rm -rf ~/.cache/huggingface/datasets/ceval___ceval-exam
# export HF_DATASETS_OFFLINE=1