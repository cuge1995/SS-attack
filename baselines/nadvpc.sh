CUDA_VISIBLE_DEVICES=2 python  attack_scripts/untargeted_nadvpc_attack.py --process_data --model=pointnet --batch_size=16
CUDA_VISIBLE_DEVICES=2 python  attack_scripts/untargeted_nadvpc_attack.py --process_data --model=pointnet2 --batch_size=16
CUDA_VISIBLE_DEVICES=2 python  attack_scripts/untargeted_nadvpc_attack.py --process_data --model=dgcnn --batch_size=16
CUDA_VISIBLE_DEVICES=2 python  attack_scripts/untargeted_nadvpc_attack.py --process_data --model=gdanet --batch_size=16
CUDA_VISIBLE_DEVICES=2 python  attack_scripts/untargeted_nadvpc_attack.py --process_data --model=pct --batch_size=16
CUDA_VISIBLE_DEVICES=2 python  attack_scripts/untargeted_nadvpc_attack.py --process_data --model=rscnn --batch_size=16
