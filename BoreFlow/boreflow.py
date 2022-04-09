import numpy as np
import pandas as pd

class BoreClass:
	def __init__(self, path_to_input_data):
		self.palaeoclimate = pd.read_csv(path_to_input_data)
	
# 	def do_something(self, foo, bar):
# 		self.new_var = do_something_on_script(foo, bar)
#
# my_analysis = BoreFlow(path_to_...)
# my_analysis.config