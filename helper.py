

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import numpy as np

from PIL import Image


for i in range(0,50):
    im = Image.open("screenshot-{}.png".format(i)) # Can be many different formats.
    pix = im.load()
    print sum(pix[330,291]),
