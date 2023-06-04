
from matplotlib import pyplot as plt
import numpy as np

w = 10
h = 10
fig = plt.figure(figsize=(8, 8))
columns = 2
rows = 2
for _ in range(10):
    for i in range(1, columns*rows +1):
        img = np.random.randint(10, size=(70,100))
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.draw()
    plt.pause(0.5)
    plt.clf()