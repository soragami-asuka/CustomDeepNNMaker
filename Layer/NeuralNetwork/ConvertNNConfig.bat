

call ConvertNNConfigToSource.exe "./FeedforwardNeuralNetwork/Config.xml" "./FeedforwardNeuralNetwork/" "FeedforwardNeuralNetwork"
call ConvertNNConfigToSource.exe "./FullyConnect_Activation/Config.xml" "./FullyConnect_Activation/" "FullyConnect_Activation"
call ConvertNNConfigToSource.exe "./FullyConnect/Config.xml" "./FullyConnect/" "FullyConnect"
call ConvertNNConfigToSource.exe "./Convolution/Config.xml" "./Convolution/" "Convolution"
call ConvertNNConfigToSource.exe "./Activation/Config.xml" "./Activation/" "Activation"
call ConvertNNConfigToSource.exe "./Dropout/Config.xml" "./Dropout/" "Dropout"
call ConvertNNConfigToSource.exe "./Pooling/Config.xml" "./Pooling/" "Pooling"
call ConvertNNConfigToSource.exe "./GlobalAveragePooling/Config.xml" "./GlobalAveragePooling/" "GlobalAveragePooling"
call ConvertNNConfigToSource.exe "./BatchNormalization/Config.xml" "./BatchNormalization/" "BatchNormalization"
call ConvertNNConfigToSource.exe "./SeparateOutput/Config.xml" "./SeparateOutput/" "SeparateOutput"
call ConvertNNConfigToSource.exe "./MergeInput/Config.xml" "./MergeInput/" "MergeInput"
call ConvertNNConfigToSource.exe "./Residual/Config.xml" "./Residual/" "Residual"


pause