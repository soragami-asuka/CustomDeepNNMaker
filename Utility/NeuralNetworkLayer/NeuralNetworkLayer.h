//==================================
// ニューラルネットワークのレイヤー管理用のUtiltiy
// ライブラリとして使う間は有効.
// ツール化後は消す予定
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__

#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Layer/NeuralNetwork/INNLayerConnectData.h"

#include<boost/filesystem.hpp>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	
	/** レイヤーDLL管理クラスの作成 */
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const boost::filesystem::wpath& libraryDirPath);
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const boost::filesystem::wpath& libraryDirPath);

	/** レイヤーデータを作成 */
	Layer::NeuralNetwork::INNLayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);
	Layer::NeuralNetwork::INNLayerData* CreateConvolutionLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize);
	Layer::NeuralNetwork::INNLayerData* CreateFullyConnectLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, U32 neuronCount);
	Layer::NeuralNetwork::INNLayerData* CreateActivationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, const std::wstring activationType);
	Layer::NeuralNetwork::INNLayerData* CreateDropoutLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, F32 rate);
	Layer::NeuralNetwork::INNLayerData* CreatePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize);
	Layer::NeuralNetwork::INNLayerData* CreateBatchNormalizationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);

	/** レイヤーをネットワークの末尾に追加する.GUIDは自動割り当て.入力データ構造、最終GUIDも更新する. */
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::NeuralNetwork::INNLayerConnectData& neuralNetwork,
		std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::NeuralNetwork::INNLayerData* pAddlayer);


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif

