import argparse
import os.path

import dill

from jax import random, tree_util

from naba import costs
from naba.data import generate_eval_data
from naba.priors import priors_default


def parse_args():
    parser = argparse.ArgumentParser(description="Generate evaluation data")
    parser.add_argument("--cost", type=str, default="QuadraticCostQuadraticEffort",
                        help="Cost function type")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--eval_size", type=int, default=100_000)
    parser.add_argument("--analytical", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    priors = priors_default

    key = random.PRNGKey(args.seed)

    cost_fn = getattr(costs, args.cost)

    # generate some data
    eval_data = generate_eval_data(key, priors=priors, cost_fn=cost_fn, num_samples=int(args.eval_size),
                                   analytical=args.analytical)
    m_test, sensorimotor_params_test, cost_params_test, a_test = eval_data

    # get valid examples only
    idx = (a_test > 0) & (a_test < 1e6)
    get_valid = lambda x: tree_util.tree_map(lambda child: child[idx], x)
    eval_data = m_test[idx], get_valid(sensorimotor_params_test), get_valid(cost_params_test), a_test[idx]

    if not os.path.exists("data/eval"):
        os.makedirs("data/eval")
    dill.dump(eval_data, open(f"data/eval/{args.cost}.pkl", "wb"))
