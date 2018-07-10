# symmetric_functions_neural_network
We show that TensorFlow DNN.classifier fails to learn symmetric polynomials:

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

