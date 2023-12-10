"""Simulation visualizer + recorder using p5 Python.

Import best FishDNA from a Generation file, visualize its behavior.

Usage: python p5_visualize.py path/to/generation_file.csv

Notes:
I used this script to verify my model worked and generate the deliverable videos
I presented. If recording, it writes each frame as a PNG to a directory, which
I converted into a video with ffmpeg. This isn't the cleanest script in the world, 
but it did what I needed it to do. 

I use p5py's experimental SKIA renderer for fast rendering without recording, 
but have to default to the slower VisPy renderer for recording since save_frame
isn't supported with SKIA. Given more time, I also would've written a config
loader to load Fish and Predator config into the visualizer, but I ended up just
manually changing the config values in the class definitions manually.
"""

# ffmpeg command to package output frames into a video:
# ffmpeg -r {60 // updates_per_frame} -f image2 -s {dims[0]xdims[1]} -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
# ffmpeg -r 30 -f image2 -s 1000x1000 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4
# ffmpeg -r 30 -f image2 -s 1000x1000 -i frame%04d.png -vcodec libx264 -crf 25 -pix_fmt yuv420p test.mp4

import sys
import csv
import numpy as np
from p5 import *
import pathlib
from agents import AgentManager, FishDna

def get_best_dna(csv_fp):
    """Extract best FishDna from a generation CSV file."""
    csv_data = []
    with open(csv_fp, 'r', newline='') as f:
        reader = csv.reader(f)
        for line in reader:
            csv_data.append(line)
    best_row = csv_data[1]
    best_dna_data = {
        'cohesion': float(best_row[0]),
        'separation': float(best_row[1]),
        'alignment': float(best_row[2]),
        'fear': float(best_row[3]),
    }
    return FishDna(data=best_dna_data)


RECORDING = True
RENDER_VISION = False
MAX_FRAMES = 1000

dims = np.array([1000, 1000])
num_fish = 400
num_predators = 3
scale_mult = 2

curr_frame = 0
updates_per_frame = 1
renderer = 'vispy' if RECORDING else 'skia'

fp = sys.argv[1]
savedir = fp.rsplit('.', 1)[0] + '_imgs/'
best_dna = get_best_dna(sys.argv[1])
manager = AgentManager(best_dna, scale_mult, dims=dims, show_vision_radius=RENDER_VISION)
manager.populate(num_fish, num_predators)

if RECORDING:
    pathlib.Path(savedir).mkdir(exist_ok=True)

def setup():
    size(*dims)

def draw():
    global curr_frame
    background(51)
    for _ in range(updates_per_frame):
        manager.update()
    manager.p5_render()
    if RECORDING:
        savefp = savedir + f'frame{curr_frame:04}.png'
        save_canvas(savefp)
    curr_frame += 1
    if (MAX_FRAMES > 0 and curr_frame > MAX_FRAMES):
        exit()

run(renderer=renderer)