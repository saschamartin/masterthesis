from PIL import Image
import glob
from natsort import natsorted
import os

path = "Export/0/run 306/"
frames = []
imgs_dist = glob.glob(path+"*_generation_dist.png")
imgs_dist = natsorted(imgs_dist)
for i in imgs_dist:
    new_frame = Image.open(i)
    frames.append(new_frame)


frames[0].save(path+"DistillationConvergence.gif",format="GIF",
               append_images=frames[1:],
               save_all=True,
               duration = 30, loop=0)

# --------------
#  PhaseDiagram
# --------------
frames = []
imgs_PD = glob.glob(path+"*_generation_PD.png")
imgs_PD = natsorted(imgs_PD)
for i in imgs_PD:
    new_frame = Image.open(i)
    frames.append(new_frame)


frames[0].save(path+"PhaseDiagram.gif",format="GIF",
               append_images=frames[1:],
               save_all=True,
               duration = 30, loop=0)