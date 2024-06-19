# Body Dataset
# CUDA_VISIBLE_DEVICES=0 python train.py data/surreal_processed data/Human36M \
#     -s SURREAL -t Human36M --seed 0 --debug --rotation 30 --epochs 10 --log logs/MBDA/surreal2human36m
# CUDA_VISIBLE_DEVICES=0 python train.py data/surreal_processed data/Human36M \
#     -s SURREAL -t Human36M --seed 0 --debug --rotation 30 --epochs 20 --log logs/MBDA/surreal2human36m1
CUDA_VISIBLE_DEVICES=4 python train.py ~/projects/datasets/SURREAL ~/projects/datasets/LSP \
    -s SURREAL -t LSP --seed 0 --debug --rotation 30 --log logs/train/surreal2lsp
