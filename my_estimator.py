#!/anaconda3/envs/tensorflow/bin/python
import tensorflow as tf
import generate_data
import argparse
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
	# eval_result = classifier.evaluate(
	# 	input_fn=lambda:iris_data.eval_input_fn(test_x, test_y,
	# 											args.batch_size))

	# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	#
	# # Generate predictions from the model
	# expected = ['Setosa', 'Versicolor', 'Virginica']
	# predict_x = {
	# 	'SepalLength': [5.1, 5.9, 6.9],
	# 	'SepalWidth': [3.3, 3.0, 3.1],
	# 	'PetalLength': [1.7, 4.2, 5.4],
	# 	'PetalWidth': [0.5, 1.5, 2.1],
	# }
	#
	# predictions = classifier.predict(
	# 	input_fn=lambda:iris_data.eval_input_fn(predict_x,
	# 											labels=None,
	# 											batch_size=args.batch_size))
	#
	# template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')
	#
	# for pred_dict, expec in zip(predictions, expected):
	# 	class_id = pred_dict['class_ids'][0]
	# 	probability = pred_dict['probabilities'][class_id]
	#
	# 	print(template.format(iris_data.SPECIES[class_id],
	# 						  100 * probability, expec))

if __name__ == '__main__':
	tf.logging.set_verbosity(tf.logging.INFO)
	tf.app.run(main)


#
