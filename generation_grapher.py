"""Visualizes the progress of a GA run from generation CSV files.

Notes:
This is another quick-and-dirty script I used to generate fitness vs. generation
plots. Given more time, I would've fleshed this out into a better script, but 
this I all I needed at the time.
"""
import sys
import csv
import matplotlib.pyplot as plt
import glob
from pathlib import Path


def read_gen(fp):
    data = []
    with open(fp, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(row)
    return data


def process_generation(generation_dir):
    template_glob = "generation_{}_*.csv"

    all_gen_data = []
    gen_best_fitness = [] 
    gen_avg_fitness = []

    num_files = len(glob.glob(generation_dir + '/' + 'generation*.csv'))

    for i in range(num_files):
        glob_pattern = generation_dir + '/' + template_glob.format(i)
        fp = glob.glob(glob_pattern)
        print(fp)
        fp = fp[0]
        data = read_gen(fp)
        data = data[1:]
        all_gen_data.append(data)

        fitness = [int(row[-1]) for row in data]
        best_fitness = max(fitness)
        avg_fitness = sum(fitness) / len(fitness)
        gen_best_fitness.append(best_fitness)
        gen_avg_fitness.append(avg_fitness)
    
    plot_generation_stats(generation_dir, gen_best_fitness, gen_avg_fitness)


def plot_generation_stats(generation_dir, gen_best_fitness, gen_avg_fitness):
    # Plot best and avg fitness vs generations
    plt.clf()
    plt.title("Fitness vs. Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")

    xs = list(range(len(gen_best_fitness)))
    plt.plot(xs, gen_best_fitness, label='Best Fitness')
    plt.plot(xs, gen_avg_fitness, label='Average Fitness')
    plt.legend()

    plt.savefig(f"{str(generation_dir)}_plot.png")


# Dependent on working directory -- this was hardcoded to whatever GA runs were in my working directory.
default_generation_dir = 'generations_3_par/'
if len(sys.argv) >= 2:
    generation_dirs = sys.argv[1:]
else:
    generation_dirs = [default_generation_dir]

generation_dirs = [
    'generation_2',
    'generation_3',
    'generation_3_par',
    'generations',
    'generations_1',
    'generations_2',
    'generations_4',
]

for generation_dir in generation_dirs:
    generation_dir = str(Path(generation_dir))
    print(f"Processing {generation_dir}:")
    process_generation(generation_dir)