from sympy.polys.polyutils import _analyze_gens
from sympy.core import Add, Mul, Symbol, sympify, Dummy, symbols
from sympy.core.singleton import S
from sympy.utilities import subsets
from sympy import symbols
from itertools import combinations
from sympy.polys.specialpolys import symmetric_poly
import random
import sys
import csv
import pandas as pd
import numpy as np
def generate_csv(filename, gens, polynoms, values):

	# lines 18-34 below get values ready to write to csv file
	# getting highest degree of polynomials in polynoms
	d = len(gens)
	# print("d={}".format(d))
	n = len(polynoms)

	csvfile = open(filename,"w")
	fieldnames = values[0].keys()
		# print(values[0].keys())
	writer  = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for i in range(34):
		writer.writerow(values[i])

	csvfile.close()


	return fieldnames
def load_csv(filename,CSV_COLUMNS):
	train = pd.read_csv("./train.csv")
	print(train)
   # train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
	return

# generating n symbols

def generate_polynoms(d, n):
	polynoms = []
# setting dimension

	gens=[]
	for i in range(1,d+1):
		print("i={}".format(i))
		x_i=symbols("x"+str(i))
		gens.append(x_i)
	# print("gens = {}".format(gens))
# print("len(gens)={}".format(len(gens)))
# generating list of n polynomials with x1,x2,...,xd arguments
# the degrees are between 1 and d
	# for _ in range(n):

	for i in range(1,d+1):
		polynoms.append(symmetric_poly(i,gens))

	return polynoms, gens

def generate_values(polynoms,gens, n):


# setting random values to arguments in gens
# number of different argument values
	d = len(gens)
	values={}
	for i in range(n):
		values[i]={}
		value={}
		for x in gens:
			value[x]=random.randint(1,10)
		k = random.randint(0,d-1)
		xd1=symbols("x"+str(d+1))
		xd2=symbols('p')
		value[xd1]=int(polynoms[k].evalf(subs=values[i]))
		value[xd2]=k
		values[i]=value

	return values
	# x1= ., x2=., ...


if __name__=="__main__":
	polynoms,gens = generate_polynoms(4,2)
	values   = generate_polynom_values(polynoms,34)
	CSV_COLUMNS = generate_csv("train.csv",gens, polynoms,values)
	load_csv("train.csv", CSV_COLUMNS)
#
