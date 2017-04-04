
set CONFIG_FILEPATH="./FeedforwardNeuralNetwork/Config.xml"
set EXPORT_DIRPATH="./FeedforwardNeuralNetwork/"
set EXPORT_NAME="FeedforwardNeuralNetwork"

call ConvertNNConfigToSource.exe %CONFIG_FILEPATH% %EXPORT_DIRPATH% %EXPORT_NAME%

pause