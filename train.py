import argparse
import os

from jax import random
import optax
import dill

from naba.nn import architecture, train, save
from naba import costs
from naba.priors import priors_default


def parse_args():
    parser = argparse.ArgumentParser(description="Train neural network")
    parser.add_argument("--cost", type=str, default="QuadraticCostQuadraticEffort",
                        help="Cost function type")
    parser.add_argument("--optimizer", type=str, default="rmsprop",
                        help="Optimizer (needs to be in optax)")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for the optimizer")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size of parameters used for training")
    parser.add_argument("--num_monte_carlo_samples", type=int, default=128,
                        help="Number of samples to evaluate the unsupervised loss.")
    parser.add_argument("--print_every", type=int, default=1_000,
                        help="Print evaluation loss every n training steps.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for training")
    parser.add_argument("--num_hidden_units", type=int, nargs="+", default=[16, 64, 16, 8],
                        help="Neural network number of hidden units")
    parser.add_argument("--architecture", type=str, default="ActionNetwork",
                        help="Neural network architecture (must be in naba.nn.architecture)")
    parser.add_argument("--eval_size", type=int, default=5_000,
                        help="Size of evaluation dataset (only used if no pre-generated dataset is found).")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    priors = priors_default

    # setup random keys
    model_key, train_key = random.split(random.PRNGKey(args.seed), 2)

    # get cost function type
    cost_fn = getattr(costs, args.cost)

    # init neural network object
    nn = getattr(architecture, args.architecture)(model_key, cost_fn=cost_fn, num_hidden_units=args.num_hidden_units)

    # setup optimizer
    optim = getattr(optax, args.optimizer)(learning_rate=args.learning_rate)

    try:
        eval_data = dill.load(open(f"data/eval/{args.cost}.pkl", "rb"))
        print("Using pre-generated evaluation data from", f"data/eval/{args.cost}.pkl")
    except FileNotFoundError:
        print("No pre-generated evaluation data found at", f"data/eval/{args.cost}.pkl.", "Generating from scratch..")

        eval_data = None

    nn = train(train_key, model=nn, priors=priors, cost_fn=cost_fn, optim=optim,
               steps=args.steps, batch_size=args.batch_size,
               print_every=args.print_every,
               eval_data=eval_data, eval_size=args.eval_size,
               evaluate=True)

    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    save(f"checkpoints/{args.cost}-{args.architecture}.pkl", model=nn)
