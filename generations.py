"""Generation definitions for assembling genetic algorithm iterations."""
import numpy as np
import csv
import logging
import multiprocessing
from agents import AgentManager, FishDna


logging.basicConfig(level=logging.DEBUG)


class GenerationManager:
    """GA-level manager for all generations in a genetic algorithm."""

    def __init__(self, num_simulations, simulation_length, num_fish, num_predators, scale, outdir):
        self.generations = []
        self.num_simulations = num_simulations
        self.simulation_length = simulation_length
        self.num_fish = num_fish
        self.num_predators = num_predators
        self.scale = scale
        self.outdir = outdir

    def run_generations(self, num_generations):
        """Advance the genetic algorithm by num_generations."""
        for i in range(num_generations):
            # Set up generation to execute 
            # Running from first generation
            if not self.generations:
                logging.info(f"Running first generation {i}")
                random_fish_dnas = [FishDna() for _ in range(self.num_simulations)]
                generation = Generation(self.simulation_length, self.scale)
                generation.populate(random_fish_dnas, self.num_fish, self.num_predators)
            # Running from completed previous generation
            else:
                assert self.generations[-1].complete
                logging.info(f"Running generation {i}")
                prev_generation = self.generations[-1]
                best_dnas, _ = prev_generation.select_best_dna_fitness()
                new_dnas = prev_generation.reproduce(best_dnas, self.num_simulations)
                generation = Generation(self.simulation_length, self.scale)
                generation.populate(new_dnas, self.num_fish, self.num_predators)
            # Execute and save newly created generation
            generation.run_all_simulations()
            csv_fp = f"{self.outdir}/generation_{len(self.generations)}_{self.simulation_length}_{self.num_fish}_{self.num_predators}.csv"
            generation.write_dna_pool(csv_fp)
            self.generations.append(generation)
            best_fitness = generation.select_best_dna_fitness(keep=1)[1]
            logging.info(f"Wrote generation to {csv_fp}: best_fitness = {best_fitness}")


class Generation:
    """State manager for all simulations in a generation."""

    def __init__(self, simulation_length, scale, fish_config=None, predator_config=None):
        self.simulation_length = simulation_length
        self.scale = scale
        self.agent_managers = []
        self.mutation_prob = 0.05
        self.mutation_std = 0.05
        self.rng = np.random.default_rng(seed=42)
        self.complete = False
        self.parallelism = 20
        self.fish_config = fish_config
        self.predator_config = predator_config

    def populate(self, fish_dnas, num_fish, num_predators):
        """Populate all simulations with appropriate DNA and agents."""
        for fish_dna in fish_dnas:
            manager = AgentManager(
                fish_dna, self.scale, rng=self.rng, 
                fish_config=self.fish_config, predator_config=self.predator_config)
            manager.populate(num_fish, num_predators)
            self.agent_managers.append(manager)

    def run_all_simulations(self):
        """Run all simulations to completion."""
        # NOTE: Subprocesses don't exit gracefully, fallback to sequential to debug
        with multiprocessing.Pool(self.parallelism) as pool:
            self.agent_managers = pool.starmap(run_simulation, [(mgr, self.simulation_length) for mgr in self.agent_managers])
        self.complete = True
        # Non-parallelized implementation of run_all_simulations:
        # for manager in self.agent_managers:
        #     logging.debug("Running %s", manager)
        #     for _ in range(self.simulation_length):
        #         manager.update()
        #     logging.debug("Completed %s", manager)
        # self.complete = True

    def select_best_dna_fitness(self, keep=None):
        """Return best FishDna and corresponding fitnesses from AgentManagers."""
        if keep is None:
            keep = len(self.agent_managers) // 2
        data = [(manager.calc_fitness(), manager.fish_dna) for manager in self.agent_managers]
        data.sort(key=lambda x: x[0], reverse=True)
        best_fitness = [datum[0] for datum in data[:keep]]
        best_dna = [datum[1] for datum in data[:keep]]
        return best_dna, best_fitness
    
    def mutate(self, dna):
        """Return mutated input FishDna according to mutation_prob and mutation_std."""
        mutated_dna_data = {}
        for k, v in dna.data.items():
            if self.rng.random() > self.mutation_prob:
                mutated_dna_data[k] = v
            else:
                mutated_dna_data[k] = v + self.rng.normal(0, self.mutation_std)
        return FishDna(data=mutated_dna_data)
    
    def reproduce(self, best_dna, desired_population_size):
        """Reproduce to desired level with crossover and mutation."""
        assert len(best_dna) >= 2
        num_reproductions = desired_population_size - len(best_dna)
        reproduced_dna = []
        for _ in range(num_reproductions):
            dna1, dna2 = self.rng.choice(best_dna, 2)
            new_dna = FishDna.uniform_crossover(dna1, dna2)
            new_dna = self.mutate(new_dna)
            reproduced_dna.append(new_dna)
        new_dna_pool = best_dna + reproduced_dna
        self.rng.shuffle(new_dna_pool)
        return new_dna_pool
    
    def write_dna_pool(self, fp):
        """Write current DNA pool to a CSV to save progress."""
        assert fp.endswith('.csv')
        table_keys = ['cohesion', 'separation', 'alignment', 'fear', 'fitness']
        table_rows = [
            [manager.fish_dna.data[dna_key] for dna_key in table_keys[:-1]] + [manager.calc_fitness()]
            for manager in self.agent_managers
        ]
        table_rows.sort(key=lambda x: x[-1], reverse=True)
        table = [table_keys] + table_rows
        with open(fp, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(table)

    @classmethod
    def import_from_csv(cls, fp, simulation_length, num_fish, num_predators):
        assert fp.endswith('.csv')
        with open(fp, newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)[1:]
            imported_dna = []
            for row in rows:
                data = {
                    'cohesion': float(row[0]),
                    'separation': float(row[1]),
                    'alignment': float(row[2]),
                    'fear': float(row[3]),
                }
                imported_dna.append(FishDna(data))
        generation = cls(simulation_length)
        generation.populate(imported_dna, num_fish, num_predators) 


# NOTE: Had to define this separately because multiprocessing can't parallelize
# a function it can't pickle, like local functions and lambda functions.
# I may be able to make this a member function, but haven't tried to because
# doing this made the code work.
def run_simulation(manager, simulation_length):
    logging.debug("Running %s", manager)
    for _ in range(simulation_length):
        manager.update()
    logging.debug("Completed %s", manager)
    return manager