import generate_data
# we transform data to feed into model
def parse_csv(line):
	example_defaults = [[0], [0], [0], [0], [0]]  # sets field types
	column_names = ['x1','x2','x4','x5','p']
	feature_names = column_names[:-1]
	label_name = column_names[-1]
	print("feature_names={}".format(feature_names))
	print("label_names={}".format(feature_names))

	print("Features: {}".format(feature_names))
	print("Label: {}".format(label_name))
	  # parsed_line = tf.decode_csv(line, example_defaults)
  # First 4 fields are features, combine into single tensor
	  # features = tf.reshape(parsed_line[:-1], shape=(4,))
  # Last field is the label
  # label = tf.reshape(parsed_line[-1], shape=())
	# return features, label

if __name__=="__main__":
# 0) Generate and save data: lines 103-105
# 1) Import and parse the data sets: lines 106-108
# 2)Create feature columns to describe the data.
# 3)Select the type of model
# 4)Train the model.
# 5)Evaluate the model's effectiveness.
# 6)Let the trained model make predictions.

	polynoms,gens = generate_data.generate_polynoms(4,2)
	values   = generate_data.generate_values(polynoms,gens, 34)
	# 0..34 = 0..24 and 25..24
	generate_data.generate_csv("train.csv",gens, polynoms,values, 0,24)
	generate_data.generate_csv("test.csv",gens, polynoms,values,25,34)
	train_features, train_label = generate_data.load_csv("./train.csv")
	test_features, test_label = generate_data.load_csv("./test.csv")
	my_feature_columns = []
	for key in train_features.keys():
		print("train_key={}".format(key))
	for key in test_features.keys():
		print("test_key={}".format(key))

#
