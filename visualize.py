import numpy as np 
import matplotlib.pyplot as plt

image = []
with open("result") as f:
    for line in f:
        image.append([int(i) for i in line.split(",")])

plt.imshow(image, interpolation="none")
plt.show()