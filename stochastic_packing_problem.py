import random
import math
from qubots.base_problem import BaseProblem

def generate_scenarios(nb_items, nb_scenarios, rng_seed):
    random.seed(rng_seed)
    # For each item, choose a random uniform distribution range.
    items_dist = []
    for _ in range(nb_items):
        item_min = random.randint(10, 100)
        item_max = item_min + random.randint(0, 50)
        items_dist.append((item_min, item_max))
    # For each scenario, sample a weight for each item.
    scenario_item_weights = [
        [random.randint(*dist) for dist in items_dist]
        for _ in range(nb_scenarios)
    ]
    return scenario_item_weights

class StochasticPackingProblem(BaseProblem):
    """
    Stochastic Packing Problem

    A set of items (numbered 0 to nb_items-1) must be partitioned into bins.
    In each scenario, each item’s weight is sampled from a uniform distribution
    (with parameters chosen randomly at instance generation). For a given partition,
    the weight of a bin in a scenario is the sum of the weights of the items assigned
    to it. The objective is to minimize the 90th percentile (9th decile) of the maximum
    bin weight over all scenarios. This robust criterion balances average performance
    with protection against extreme cases.

    Decision Variables:
      A candidate solution is represented as a list of bins (one per bin).
      Each bin is a list of item indices.
      Every item must appear in exactly one bin.
    """
    def __init__(self, nb_items, nb_bins, nb_scenarios, rng_seed):
        # Instead of reading an instance file, we use default parameters.
        # These parameters can be overridden via the override_params mechanism.
        self.nb_items = nb_items
        self.nb_bins = nb_bins
        self.nb_scenarios = nb_scenarios
        self.rng_seed = rng_seed
        # Generate the scenario data: a list (over scenarios) of lists (item weights)
        self.scenario_item_weights_data = generate_scenarios(
            self.nb_items, self.nb_scenarios, self.rng_seed
        )

    def evaluate_solution(self, candidate) -> float:
        penalty = 0
        # Check partition validity: candidate should be a partition of {0,...,nb_items-1}.
        assigned_items = []
        for bin in candidate:
            assigned_items.extend(bin)
        if sorted(assigned_items) != list(range(self.nb_items)):
            penalty += 1e6

        scenario_max_weights = []
        # For each scenario, compute the weight of each bin and record the maximum.
        for s in range(self.nb_scenarios):
            bin_weights = []
            for bin in candidate:
                weight = sum(self.scenario_item_weights_data[s][i] for i in bin)
                bin_weights.append(weight)
            # In case a bin is empty, its weight is 0.
            scenario_max_weights.append(max(bin_weights) if bin_weights else 0)
        # Sort the maximum weights and pick the 9th decile.
        sorted_weights = sorted(scenario_max_weights)
        index = int(math.ceil(0.9 * (self.nb_scenarios - 1)))
        objective_value = sorted_weights[index]
        return objective_value + penalty

    def random_solution(self):
        """
        Generates a random candidate solution by randomly assigning each item
        to one of the available bins.
        """
        items = list(range(self.nb_items))
        random.shuffle(items)
        # Randomly assign each item to a bin.
        assignment = [random.randint(0, self.nb_bins - 1) for _ in range(self.nb_items)]
        bins = [[] for _ in range(self.nb_bins)]
        for item, bin_id in zip(items, assignment):
            bins[bin_id].append(item)
        return bins
