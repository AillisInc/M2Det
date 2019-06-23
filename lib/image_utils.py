import matplotlib.pyplot as plt
import numpy as np
import torchvision

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def imshow_grid(images):
    imshow(torchvision.utils.make_grid(images))