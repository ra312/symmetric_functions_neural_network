#!/anaconda3/envs/tensorflow/bin/python
import tensorflow as tf
import generate_data
import argparse
import random
import numpy.random as np_random
from sympy import symbols
# we transform data to feed into model
def parse_csv(line):
	example_defaults = [[0], [0], [0], [0], [0]]  # sets field types
	column_names = ['x1','x2','x4','x5','p']
	feature_names = column_names[:-1]
	label_name = column_names[-1]
	# print("feature_names={}".format(feature_names))
	# print("label_names={}".format(feature_names))

	print("Features: {}".format(feature_names))
	print("Label: {}".format(label_name))
	  # parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
	  # features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  # label = tf.reshape(parsed_line[-1], shape=())
	# return features, label

# 0) Generate and save data: lines 103-105

# 1) Import and parse the data sets: lines 106-108
# 2)Create feature columns to describe the data.
# 3)Select the type of model
# 4)Train the model.
# 5)Evaluate the model's effectiveness.
# 6)Let the trained model make predictions.

def main(argv):
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch_size', default=100, type=int, help='batch size')
	parser.add_argument('--train_steps', default=1000, type=int,
					help='number of training steps')
	args = parser.parse_args(argv[1:])
	# print("args={}".format(args))
	# Fetch the data
	polynoms,gens = generate_data.generate_polynoms(4,2)
	values   = generate_data.generate_values(polynoms,gens, 34)
	# 0..34 = 0..24 and 25..24
	generate_data.generate_csv("train.csv",gens, polynoms,values, 0,24)
	generate_data.generate_csv("test.csv",gens, polynoms,values,25,34)
	train_features, train_label = generate_data.load_csv("./train.csv")
	test_features, test_label = generate_data.load_csv("./test.csv")
	# Feature columns describe how to use the input.
	my_feature_columns = []
	for key in train_features.keys():
		# print("key={}".format(key))
		my_feature_columns.append(tf.feature_column.numeric_column(key=key))
		# print("train_key={}".format(key))
	dataset = generate_data.train_input_fn(train_features,train_label,args.batch_size)
	# print("dataset={}".format(dataset))
# Build 2 hidden layer DNN with 10, 10 units respectively.
	classifier = tf.estimator.DNNClassifier(
		feature_columns=my_feature_columns,
		# Two hidden layers of 10 nodes each.
		hidden_units=[10, 10],
		# The model must choose between 3 classes.
		n_classes=4)

	# Train the Model.
	classifier.train(
		  input_fn=lambda:generate_data.train_input_fn(train_features, train_label,
												   args.batch_size),
		  steps=args.train_steps)

	# Evaluate the model.
	eval_result = classifier.evaluate(
		 input_fn=lambda:generate_data.eval_input_fn(test_features, test_label,
												 args.batch_size))

	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	#
	# # Generate predictions from the model

	predict_value = {}

	x1=symbols('x1')
	x2=symbols('x2')
	x3=symbols('x3')
	x4=symbols('x4')
	x5=symbols('x5')
	d = len(gens)
	predict_x = {}
	# for x in gens:
	# 	 predict_x[x]=np_random.randint(1,10, size=d)
	predict_x = {
		'x1'	:	[3, 4, 9, 6],
		'x2'	:	[1, 1, 5, 4],
		'x3'	:	[1, 7, 7, 8],
		'x4'	:	[7, 3, 9, 1]
	}
	# print(predict_x)
	expected=['0','1','2','3']
	p0 = {'x'+str(k):predict_x['x'+str(k)][0] for k in range(1,5)}
	p1 = {'x'+str(k):predict_x['x'+str(k)][1] for k in range(1,5)}
	p2 = {'x'+str(k):predict_x['x'+str(k)][2] for k in range(1,5)}
	p3 = {'x'+str(k):predict_x['x'+str(k)][3] for k in range(1,5)}
	p4 = []
	p4.append(
				int(polynoms[0].evalf(subs=p0))
			 )
	p4.append(
				int(polynoms[1].evalf(subs=p1))
			 )
	p4.append(
				int(polynoms[2].evalf(subs=p2))
			 )
	p4.append(
				int(polynoms[3].evalf(subs=p3))
			 )
	predict_x['x5']=p4
	# print(predict_x)
	# print(p0)
	# print(p1)
	# print(p2)
	# print(p3)
	# print(p4)




	# 		predict_x[x5]=int(polynoms[k].evalf(subs=predict_x[x]))

	# d = len(gens)
	# xd1=symbols("x"+str(d+1))
	# xd2=symbols('p')
	# for k in range(d):
	# 	predict_poly_k_eval = int(polynoms[k].evalf(subs=predict_value))
	# 	predict_value[xd1]= predict_poly_k_eval
	#
	# expected = [0, 1, 2, 3]


	#
	predictions = classifier.predict(
		  input_fn=lambda:generate_data.eval_input_fn(predict_x,
												  labels=None,
												  batch_size=args.batch_size)
	)
	template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
	for pred_dict, expec in zip(predictions, expected):
		class_id = pred_dict['class_ids'][0]
		probability = pred_dict['probabilities'][class_id]
		print(template.format(generate_data.SPECIES[class_id],
						  100 * probability, expec))


if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)


#
