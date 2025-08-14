import os, time, argparse, logging, random, re
import numpy as np
import torch

from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn, TimeRemainingColumn

from models import (
    ModelConfig,
    Transformer,
    InfiniteDataLoader,
    evaluate,
)

from utils import TopPool

from training_tools import (
    setup_logging,
    set_device,
    create_datasets_from_file,
    generate_samples,
    decode_and_update_pool,
)
# --------------------------- Argparser ---------------------------------------
def get_parser():
    parser = argparse.ArgumentParser("PatternBoost with Top-Pool for no-three-in-line")

    # problem setup ----------------------------------------------------------
    parser.add_argument("--grid_size", type=int, default=6)
    parser.add_argument("--max_points", type=int, default=18)

    # data / pool ------------------------------------------------------------
    parser.add_argument("--target_training_size", type=int, default=20000,
                        help="Desired size of the pooled training set (capacity of heap)")

    # generation / selection -------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=500)
    parser.add_argument("--keep_best_fraction", type=float, default=1.0)
    parser.add_argument("--symmetrize", action=argparse.BooleanOptionalAction, default=True,
                        help="Generate all 8 symmetries of selected constructions (default on)")

    # model & optimisation ---------------------------------------------------
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=64)
    parser.add_argument("--n_embd2", type=int, default=16)

    parser.add_argument("--nn_batch_size", "-b", type=int, default=64)
    parser.add_argument("--learning_rate", "-l", type=float, default=1e-4)
    parser.add_argument("--weight_decay", "-w", type=float, default=0.1)
    parser.add_argument("--num_workers", "-n", type=int, default=8)

    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--max_epochs", type=int, default=20)

    # sampling ---------------------------------------------------------------
    parser.add_argument("--gen_batch_size", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=-1)

    # system / paths ---------------------------------------------------------
    parser.add_argument("--dump_path", type=str, default="dump_path")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=-1)

    return parser


# --------------------------- Main -------------------------------------------


if __name__ == "__main__":
    start = time.time()
    parser = get_parser()
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(__file__))
    args.dump_path = os.path.join("training", args.dump_path)
    logger = setup_logging(args.dump_path)
    set_device(args, logger)

    # seed -------------------------------------------------------------------
    if args.seed < 0:
        args.seed = np.random.randint(1_000_000_000)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    logger.info(f"seed: {args.seed}")

    # token maps -------------------------------------------------------------
    N = args.grid_size
    logger.info("Creating token maps...")
    token_encoding = torch.arange(N * N).view(N, N)
    coords = torch.stack(torch.meshgrid(torch.arange(N), torch.arange(N), indexing="xy"), -1)
    token_decoding = coords.view(-1, 2)

    # determine initial generation ------------------------------------------
    ts_dir = os.path.join(args.dump_path, "training_sets")
    gen_re = re.compile(rf"N{N}_gen(\d+)\.txt$")
    gens = [int(m.group(1)) for fn in os.listdir(ts_dir) if (m := gen_re.match(fn))]
    if not gens:
        logger.error("No generation files found. Run generate_initial.py first.")
        exit(1)
    initial_gen = max(gens)
    logger.info(f"Resuming from generation {initial_gen}")

    # build initial pool -----------------------------------------------------
    pool = TopPool(args.target_training_size, args.grid_size)
    init_file = os.path.join(ts_dir, f"N{N}_gen{initial_gen}.txt")
    pool.build_from_file(init_file)
    logger.info(f"Initial pool size: {len(pool)}")

    best_loss = None
    end_gen = initial_gen + args.max_epochs
    for gen in range(initial_gen, end_gen):
        logger.info(f"=========== Start of generation {gen + 1} ===========")

        # datasets: Load constructions and optionally create symmetrized versions on the fly
        input_gen_file = os.path.join(ts_dir, f"N{N}_gen{gen}.txt")
        train_ds, test_ds = create_datasets_from_file(input_gen_file, args, token_decoding, token_encoding, logger)

        config = ModelConfig(
            vocab_size=train_ds.get_vocab_size(),
            block_size=train_ds.get_output_length(),
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            n_embd2=args.n_embd2,
        )
        model = Transformer(config).to(args.device)
        model_path = os.path.join(args.dump_path, "model.pt")
        if os.path.isfile(model_path):
            model.load_state_dict(torch.load(model_path))
            logger.info("Resumed existing model.")

            # If resuming, evaluate the model to set the initial best_loss
            if best_loss is None:
                logger.info("Evaluating resumed model to set initial best_loss...")
                te_loss = evaluate(model, test_ds, args.device, max_batches=20)
                logger.info(f"Initial best_loss set to {te_loss:.4f}")
                best_loss = te_loss
        
        optim = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        # disable pin_memory on MPS to avoid unsupported pinning
        use_pin_memory = args.device != 'mps'
        loader = InfiniteDataLoader(train_ds, batch_size=args.nn_batch_size, pin_memory=use_pin_memory, num_workers=args.num_workers)

        model.train()  # Ensure model is in training mode
        for step in range(args.max_steps + 1):
            X, Y = [t.to(args.device) for t in next(loader)]
            logits, loss = model(X, Y)
            model.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            if step % 100 == 0:
                logger.info(f"step {step} | loss {loss.item():.4f}")
            if step and step % 500 == 0:
                tr_loss = evaluate(model, train_ds, args.device, max_batches=10)
                te_loss = evaluate(model, test_ds, args.device, max_batches=10)
                logger.info(f"step {step} train {tr_loss:.4f} test {te_loss:.4f}")
                if best_loss is None or te_loss < best_loss:
                    logger.info(f"test loss {te_loss} is best so far, saving model")
                    torch.save(model.state_dict(), model_path)
                    best_loss = te_loss
        loader.shutdown()

        # sample -------------------------------------------------------------
        total_to_generate = int(args.target_training_size * (1 / args.keep_best_fraction))
        
        all_samples = []
        with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeRemainingColumn()) as prog:
            t = prog.add_task("Generating samples", total=total_to_generate)
            done = 0
            while done < total_to_generate:
                n = min(args.gen_batch_size, total_to_generate - done)
                model.eval()
                new_samples = generate_samples(model, train_ds, args, num_samples=n)
                model.train()
                all_samples.extend(new_samples)
                done += n
                prog.update(t, advance=n)

        # decode / update pool ----------------------------------------------
        if args.device in {"mps", "cuda"}:
            if args.device == "mps": torch.mps.empty_cache()
            elif args.device == "cuda": torch.cuda.empty_cache()
        decode_and_update_pool(args, token_decoding, token_encoding, gen + 1, logger, pool, all_samples)

        logger.info(f"=========== End of generation {gen + 1} ===========")

    logger.info("All generations completed!")
    logger.info(f"Total time: {time.time() - start:.2f} seconds")