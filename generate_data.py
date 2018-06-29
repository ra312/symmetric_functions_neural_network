from sympy import symbols
from sympy.polys.specialpolys import symmetric_poly

import random
import csv
import pandas as pd
import tensorflow as tf


def generate_csv(filename, gens, polynoms, values, a,b):

	csvfile = open(filename,"w")
	fieldnames = values[0].keys()
		# print(values[0].keys())
	writer  = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	for i in range(a,b):
		# print("i={}".format(i))
		writer.writerow(values[i])

	csvfile.close()


	return fieldnames

def load_csv(filename):
	data = pd.read_csv(filename)
	# print(train)
	# print(train.pop('p'))
	# print("CSV_COLUMNS={}".format(CSV_COLUMNS))

	data_features, data_label = data, data.pop('p')
	# print("train features")
	# print(data_features)
	# print("train label")
	# print(data_label)
   # train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)
	return data_features, data_label


# generating n symbols

def generate_polynoms(d, n):
	polynoms = []
	gens=[]
	for i in range(1,d+1):
		# print("i={}".format(i))
		x_i=symbols("x"+str(i))
		gens.append(x_i)
	for i in range(1,d+1):
		polynoms.append(symmetric_poly(i,gens))
	return polynoms, gens
# setting random values to arguments in gens
# evaluating polynomial values
# number of different argument values
def generate_values(polynoms,gens, n):
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
		poly_k_eval = int(polynoms[k].evalf(subs=value))
		value[xd1]= poly_k_eval
		value[xd2]=k
		values[i]=value

	return values

def train_input_fn(features, labels, batch_size):
	"""An input function for training"""
	# Convert the inputs to a Dataset.
	dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
	print("features={}".format(features))
	print("dict(features)={}".format(dict(features)))
	print("labels={}".format(labels))
	# Shuffle, repeat, and batch the examples.
	dataset = dataset.shuffle(1000).repeat().batch(batch_size)

	# Return the dataset.
	return dataset
