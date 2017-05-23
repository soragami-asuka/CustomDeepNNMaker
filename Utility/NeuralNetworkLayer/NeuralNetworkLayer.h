//==================================
// �j���[�����l�b�g���[�N�̃��C���[�Ǘ��p��Utiltiy
// ���C�u�����Ƃ��Ďg���Ԃ͗L��.
// �c�[������͏����\��
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__

#include"Library/NeuralNetwork/LayerDLLManager/LayerDLLManager.h"
#include"Layer/Connect/ILayerConnectData.h"

#include<boost/filesystem.hpp>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	
	/** ���C���[DLL�Ǘ��N���X�̍쐬 */
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const boost::filesystem::wpath& libraryDirPath);
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const boost::filesystem::wpath& libraryDirPath);

	/** ���C���[�f�[�^���쐬 */
	Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);
	Layer::ILayerData* CreateConvolutionLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize);
	Layer::ILayerData* CreateFullyConnectLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, U32 neuronCount);
	Layer::ILayerData* CreateActivationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, const std::wstring activationType);
	Layer::ILayerData* CreateDropoutLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, F32 rate);
	Layer::ILayerData* CreatePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, Vector3D<S32> stride);
	Layer::ILayerData* CreateBatchNormalizationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);

	Layer::ILayerData* CreateMergeInputLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct lpInputDataStruct[], U32 inputDataCount);
	Layer::ILayerData* CreateMergeInputLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const std::vector<IODataStruct>& lpInputDataStruct);
	template<typename... Rest>
	Layer::ILayerData* CreateMergeInputLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, std::vector<IODataStruct>& lpInputDataStruct, const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateMergeInputLayer(layerDLLManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}
	template<typename... Rest>
	Layer::ILayerData* CreateMergeInputLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		std::vector<IODataStruct> lpInputDataStruct;
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateMergeInputLayer(layerDLLManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}

	Layer::ILayerData* CreateResidualLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct lpInputDataStruct[], U32 inputDataCount);
	Layer::ILayerData* CreateResidualLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const std::vector<IODataStruct>& lpInputDataStruct);
	template<typename... Rest>
	Layer::ILayerData* CreateResidualLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, std::vector<IODataStruct>& lpInputDataStruct, const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateResidualLayer(layerDLLManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}
	template<typename... Rest>
	Layer::ILayerData* CreateResidualLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		std::vector<IODataStruct> lpInputDataStruct;
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateResidualLayer(layerDLLManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}


	/** ���C���[���l�b�g���[�N�̖����ɒǉ�����.GUID�͎������蓖��.���̓f�[�^�\���A�ŏIGUID���X�V����. */
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer);

	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer,
		const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount);
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer,
		const std::vector<Gravisbell::GUID>& lpInputLayerGUID);
	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer,
		std::vector<Gravisbell::GUID>& lpInputLayerGUID,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		lpInputLayerGUID.push_back(lastLayerGUID_first);
	
		return AddLayerToNetworkLast(neuralNetwork, lppLayerData, inputDataStruct, lastLayerGUID, pAddlayer, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}
	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		std::list<Layer::ILayerData*>& lppLayerData, Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		std::vector<Gravisbell::GUID> lpInputLayerGUID;
		lpInputLayerGUID.push_back(lastLayerGUID_first);

		return AddLayerToNetworkLast(neuralNetwork, lppLayerData, inputDataStruct, lastLayerGUID, pAddlayer, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}


	/** �j���[�����l�b�g���[�N���o�C�i���t�@�C���ɕۑ����� */
	Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const boost::filesystem::path& filePath);
	/** �j���[�����l�b�g���[�N���o�C�i���t�@�C������ǂݍ��ނ��� */
	Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const boost::filesystem::path& filePath);


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif

