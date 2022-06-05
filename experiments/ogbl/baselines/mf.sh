# run matrix factorization on ogbl-collab dataset

# ogbl-collab
for SEED in 1 2 3 4 5; do
  python seal_link_pred.py --dataset ogbl-collab --epochs 50 --train_mf --hidden_channels 256 --lr 0.01 --seed $SEED --batch_size 65536
done
