

call ConvertNNConfigToSource.exe "./FeedforwardNeuralNetwork/Config.xml" "./FeedforwardNeuralNetwork/" "FeedforwardNeuralNetwork"
call ConvertNNConfigToSource.exe "./FullyConnect_Activation/Config.xml" "./FullyConnect_Activation/" "FullyConnect_Activation"
call ConvertNNConfigToSource.exe "./FullyConnect/Config.xml" "./FullyConnect/" "FullyConnect"
call ConvertNNConfigToSource.exe "./Convolution/Config.xml" "./Convolution/" "Convolution"
call ConvertNNConfigToSource.exe "./Activation/Config.xml" "./Activation/" "Activation"
call ConvertNNConfigToSource.exe "./Pooling/Config.xml" "./Pooling/" "Pooling"
call ConvertNNConfigToSource.exe "./BatchNormalization/Config.xml" "./BatchNormalization/" "BatchNormalization"


pause