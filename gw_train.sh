source .venv/bin/activate

uv run no_three_in_line/gw_train_heap.py \
    --grid_size 6 \
    --max_points 12 \
    --target_training_size 500 \
    --keep_best_fraction 0.1 \
    --n_layer 8 \
    --n_head 8 \
    --n_embd 256 \
    --n_embd2 64 \
    --nn_batch_size 256 \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --num_workers 8 \
    --max_steps 1000 \
    --max_epochs 10 \
    --w-imitate 0.4 \
    --w-goal 0.6 \
    --w-len 1.0 \
    --w-invalid 1.5 \
    --w-over 2.0 \
    --w-empty 10.0 \
    --w-perfect 10.0 \
    --temperature 1.8 \
    --dump_path "rl_2" \
    --device "cpu"