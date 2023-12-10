"""Driver file for the full genetic algorithm."""
import pathlib
import logging
import sys
from generations import GenerationManager

# NOTE: This is probably outdated, I just needed to put this somewhere since I didn't implement Fish/Predator config read/write
# Implicit Default Parameters of Interest:
# - Board dimensions: 1000x1000
# - Initial random FishDna initialization (mu, sigma): (0, 0.5)
# - Separation threshold: fish.vision_rad // 2
# - Agent radius: 7px
# - Predator class variables:
#     vision_rad = 300
#     min_speed = 1
#     max_speed = 10
#     max_turn_angle = 0.15
#     random_strength = 1
# - Fish class variables:
#     vision_rad = 80
#     min_speed = 2
#     max_speed = 8
#     random_strength = 1

logging.basicConfig(level=logging.DEBUG)

generations = 15
num_simulations = 20
simulation_length = 400

num_fish = 100
num_predators = 2
scale = 2

outdir = sys.argv[1] if len(sys.argv) >= 2 else 'generations'
pathlib.Path(outdir + '/').mkdir(exist_ok=True)

if __name__ == '__main__':
    generation_manager = GenerationManager(
        num_simulations, simulation_length, num_fish, num_predators, scale, outdir)
    generation_manager.run_generations(generations)
