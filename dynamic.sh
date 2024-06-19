# Body Dataset
# CUDA_VISIBLE_DEVICES=0 python train.py data/surreal_processed data/Human36M \
#     -s SURREAL -t Human36M --seed 0 --debug --rotation 30 --epochs 10 --log logs/train/surreal2human36m
# CUDA_VISIBLE_DEVICES=0 python train.py data/surreal_processed data/Human36M \
#     -s SURREAL -t Human36M --seed 0 --debug --rotation 30 --epochs 20 --log logs/train/surreal2human36m1
CUDA_VISIBLE_DEVICES=4 python train_new_dynamic.py ~/projects/datasets/SURREAL ~/projects/datasets/LSP \
    -s SURREAL -t LSP --seed 0 --debug --rotation 30 --num-advs 1 --fgen-method 3 --fgen-param 0 --branch-tradeoffs 1 --log logs/MBDA/surreal2lsp --pretrain /home/huangyuchao/projects/projects/backup/kptd/logs/MBDA/surreal2lsp/checkpoints/pretrain.pth


# CUDA_VISIBLE_DEVICES=1 python train_new.py ~/projects/datasets/SURREAL ~/projects/datasets/LSP \
    # -s SURREAL -t LSP --seed 0 --debug --rotation 30 --num-advs 1 --log logs/MBDA/surreal2lsp

# CUDA_VISIBLE_DEVICES=1 python train_new.py ~/projects/datasets/SURREAL ~/projects/datasets/LSP \
    # -s SURREAL -t LSP --seed 0 --debug --rotation 30 --num-advs 1 --log logs/MBDA/surreal2lsp



# fgen_method = ['', 'Halfbody', 'RandomChoice', 'Dynamic']
#         self.body_part_index = []       # 0 : upper,   1 : lower, 2: left, 3: right
#         self.body_part_index.append((6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
#         self.body_part_index.append((0, 1, 2, 3, 4, 5))
#         self.body_part_index.append((3, 4, 5, 6, 7, 8, 9, 13, 14, 15))
#         self.body_part_index.append((0, 1, 2, 6, 7, 8, 9, 10, 11, 12))