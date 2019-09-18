import numpy as np 
import pandas as pd 
from data_funcs import isfloat
from data_funcs import to_float


# ===================================================================
# Tests for isfloat()
# ===================================================================

def test_isfloat1():
	"""
	Tests case 1 : A float string can be cast to float
	"""
	string = "0.01"
	a_float = isfloat(string)	
	assert a_float

def test_isfloat2():
	"""
	Tests case 2: A string cannot be cast to float
	"""
	string = "0.01x"
	a_float = isfloat(string)	
	assert not a_float

def test_isfloat3():
	"""
	Tests case 3: An integer string can be cast to float
	"""
	string = "42"
	a_float = isfloat(string)	
	assert a_float

def test_isfloat4():
	"""
	Test case 4: An alphanumeric string cannot be float
	"""
	string = "42a"
	a_float = isfloat(string)	
	assert not a_float	


# ===================================================================
# Figure out which columns to use
# ===================================================================
# ===================================================================
# Figure out which columns to use
# ===================================================================
# ===================================================================
# Figure out which columns to use
# ===================================================================
# ===================================================================
# Figure out which columns to use
# ===================================================================
# ===================================================================
# Figure out which columns to use
# ===================================================================
# ===================================================================
# Figure out which columns to use
# ===================================================================
# ===================================================================
# Figure out which columns to use
# ===================================================================