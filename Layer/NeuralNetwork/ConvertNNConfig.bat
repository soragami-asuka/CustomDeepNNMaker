﻿

call ConvertNNConfigToSource.exe "./FeedforwardNeuralNetwork/Config.xml" "./FeedforwardNeuralNetwork/" "FeedforwardNeuralNetwork"
call ConvertNNConfigToSource.exe "./FullyConnect_Activation/Config.xml" "./FullyConnect_Activation/" "FullyConnect_Activation"
call ConvertNNConfigToSource.exe "./Convolution/Config.xml" "./Convolution/" "Convolution"
call ConvertNNConfigToSource.exe "./Activation/Config.xml" "./Activation/" "Activation"


pause