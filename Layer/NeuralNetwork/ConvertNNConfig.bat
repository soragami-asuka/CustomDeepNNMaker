﻿

call ConvertNNConfigToSource.exe "./FeedforwardNeuralNetwork/Config.xml" "./FeedforwardNeuralNetwork/" "FeedforwardNeuralNetwork"
call ConvertNNConfigToSource.exe "./FullyConnect_Activation/Config.xml" "./FullyConnect_Activation/" "FullyConnect_Activation"
call ConvertNNConfigToSource.exe "./FullyConnect/Config.xml" "./FullyConnect/" "FullyConnect"
call ConvertNNConfigToSource.exe "./Convolution/Config.xml" "./Convolution/" "Convolution"
call ConvertNNConfigToSource.exe "./UpConvolution/Config.xml" "./UpConvolution/" "UpConvolution"
call ConvertNNConfigToSource.exe "./UpSampling/Config.xml" "./UpSampling/" "UpSampling"
call ConvertNNConfigToSource.exe "./Activation/Config.xml" "./Activation/" "Activation"
call ConvertNNConfigToSource.exe "./Activation_Discriminator/Config.xml" "./Activation_Discriminator/" "Activation_Discriminator"
call ConvertNNConfigToSource.exe "./Dropout/Config.xml" "./Dropout/" "Dropout"
call ConvertNNConfigToSource.exe "./Pooling/Config.xml" "./Pooling/" "Pooling"
call ConvertNNConfigToSource.exe "./GlobalAveragePooling/Config.xml" "./GlobalAveragePooling/" "GlobalAveragePooling"
call ConvertNNConfigToSource.exe "./BatchNormalization/Config.xml" "./BatchNormalization/" "BatchNormalization"
call ConvertNNConfigToSource.exe "./BatchNormalizationAll/Config.xml" "./BatchNormalizationAll/" "BatchNormalizationAll"
call ConvertNNConfigToSource.exe "./Normalization_Scale/Config.xml" "./Normalization_Scale/" "Normalization_Scale"
call ConvertNNConfigToSource.exe "./SeparateOutput/Config.xml" "./SeparateOutput/" "SeparateOutput"
call ConvertNNConfigToSource.exe "./ChooseChannel/Config.xml" "./ChooseChannel/" "ChooseChannel"
call ConvertNNConfigToSource.exe "./Reshape/Config.xml" "./Reshape/" "Reshape"
call ConvertNNConfigToSource.exe "./Reshape_MirrorX/Config.xml" "./Reshape_MirrorX/" "Reshape_MirrorX"
call ConvertNNConfigToSource.exe "./Reshape_SquaresCenterCross/Config.xml" "./Reshape_SquaresCenterCross/" "Reshape_SquaresCenterCross"
call ConvertNNConfigToSource.exe "./Reshape_SquaresZeroSideLeftTop/Config.xml" "./Reshape_SquaresZeroSideLeftTop/" "Reshape_SquaresZeroSideLeftTop"
call ConvertNNConfigToSource.exe "./MergeInput/Config.xml" "./MergeInput/" "MergeInput"
call ConvertNNConfigToSource.exe "./MergeAdd/Config.xml" "./MergeAdd/" "MergeAdd"
call ConvertNNConfigToSource.exe "./Residual/Config.xml" "./Residual/" "Residual"
call ConvertNNConfigToSource.exe "./GaussianNoise/Config.xml" "./GaussianNoise/" "GaussianNoise"


pause