//==================================
// �j���[�����l�b�g���[�N�̃��C���[�Ǘ��p��Utiltiy
// ���C�u�����Ƃ��Ďg���Ԃ͗L��.
// �c�[������͏����\��
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__

#include"../Layer/NeuralNetwork/ILayerDLLManager.h"
#include"../Layer/Connect/ILayerConnectData.h"

#include<boost/filesystem.hpp>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	
	/** ���C���[DLL�Ǘ��N���X�̍쐬(CPU�p) */
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const boost::filesystem::wpath& libraryDirPath);
	/** ���C���[DLL�Ǘ��N���X�̍쐬(GPU�p) */
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const boost::filesystem::wpath& libraryDirPath);

	//====================================
	// ���C���[�f�[�^���쐬
	//====================================
	/** �����j���[�����l�b�g���[�N.
		@param layerDLLManager	���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct	���̓f�[�^�\��. */
	Layer::Connect::ILayerConnectData* CreateNeuralNetwork(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);
	/** �􍞂݃j���[�����l�b�g���[�N���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	filterSize			�t�B���^�T�C�Y.
		@param	outputChannelCount	�t�B���^�̌�.
		@param	stride				�t�B���^�̈ړ���.
		@param	paddingSize			�p�f�B���O�T�C�Y. */
	Layer::ILayerData* CreateConvolutionLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize);
	/** �S�����j���[�����l�b�g���[�N���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	neuronCount			�j���[������. */
	Layer::ILayerData* CreateFullyConnectLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, U32 neuronCount);
	/** ���������C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	activationType		���������. */
	Layer::ILayerData* CreateActivationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, const std::wstring activationType);
	/** �h���b�v�A�E�g���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	rate				�h���b�v�A�E�g��.(0.0�`1.0).(0.0���h���b�v�A�E�g�Ȃ�,1.0=�S���͖���) */
	Layer::ILayerData* CreateDropoutLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, F32 rate);
	/** �v�[�����O���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	filterSize			�v�[�����O��.
		@param	stride				�t�B���^�ړ���. */
	Layer::ILayerData* CreatePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, Vector3D<S32> stride);
	/** �o�b�`���K�����C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	Layer::ILayerData* CreateBatchNormalizationLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);
	/** �L�敽�σv�[�����O���C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	Layer::ILayerData* CreateGlobalAveragePoolingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct);
	/** �A�b�v�T���v�����O���C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	upScale				�g����.
		@param	paddingUseValue		�g�������̌����߂ɗאڂ���l���g�p����t���O. (true=UpConvolution, false=TransposeConvolution) */
	Layer::ILayerData* CreateUpSamplingLayer(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, const IODataStruct& inputDataStruct, Vector3D<S32> upScale, bool paddingUseValue);

	/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	inputDataCount		���͂���郌�C���[�̌�. */
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


	/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	inputDataCount		���͂���郌�C���[�̌�. */
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

