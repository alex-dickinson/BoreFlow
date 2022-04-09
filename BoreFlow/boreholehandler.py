# import numpy as np
# import pandas as pd
# import dataclasses

from BoreFlow import general_python_functions

class BoreHole:
	def __init__(self, cfg):
		self.config = cfg['borehole']
		self.data = general_python_functions.EmptyClass()

