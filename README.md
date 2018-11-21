# symmetric_functions_neural_network
We show that TensorFlow DNN.classifier fails to learn symmetric polynomials:
To run the script I used conda package manager to activate tensorflow environment.
Below is the information about the version of conda I used

	 active environment : tensorflow
		active env location : /anaconda3/envs/tensorflow
						shell level : 1
			 user config file : /Users/ra312/.condarc
 populated config files : /Users/ra312/.condarc
					conda version : 4.5.11
		conda-build version : 3.4.1
				 python version : 3.6.4.final.0
			 base environment : /anaconda3  (writable)
					 channel URLs : https://repo.anaconda.com/pkgs/main/osx-64
													https://repo.anaconda.com/pkgs/main/noarch
													https://repo.anaconda.com/pkgs/free/osx-64
													https://repo.anaconda.com/pkgs/free/noarch
													https://repo.anaconda.com/pkgs/r/osx-64
													https://repo.anaconda.com/pkgs/r/noarch
													https://repo.anaconda.com/pkgs/pro/osx-64
													https://repo.anaconda.com/pkgs/pro/noarch
					package cache : /anaconda3/pkgs
													/Users/ra312/.conda/pkgs
			 envs directories : /anaconda3/envs
													/Users/ra312/.conda/envs
							 platform : osx-64
						 user-agent : conda/4.5.11 requests/2.18.4 CPython/3.6.4 Darwin/17.7.0 OSX/10.13.6
								UID:GID : 501:20
						 netrc file : None
					 offline mode : False

Creating a tensorflow environment in conda
	conda create -n tensorflow
You can use either conda or pip to install TensorFlow.

Running the script

python my_estimator.py

Test set accuracy: 0.788

Prediction is "0" (99.7%), expected "0"

Prediction is "3" (53.6%), expected "1"

Prediction is "2" (54.1%), expected "2"

Prediction is "2" (39.4%), expected "3"

Labels description:
"0" -> x1 + x2 + x3 + x4

"1" ->x1*x2 + x1*x3 + x1*x4 + x2*x3 + x2*x4 + x3*x4

"2" -> x1*x2*x3 + x1*x2*x4 + x1*x3*x4 + x2*x3*x4

"3" -> x1*x2*x3*x4
