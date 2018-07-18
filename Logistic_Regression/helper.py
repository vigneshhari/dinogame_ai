

from selenium import webdriver
from selenium.webdriver.common.keys import Keys

import numpy as np

from PIL import Image


import pandas

l = [1,2,3,4]
ll = [5,6,7,8]


df = pandas.DataFrame(data={"col1": l, "col2": ll})
df.to_csv("file.csv", sep=',',index=False)