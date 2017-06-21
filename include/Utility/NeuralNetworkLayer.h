//==================================
// �j���[�����l�b�g���[�N�̃��C���[�Ǘ��p��Utiltiy
// ���C�u�����Ƃ��Ďg���Ԃ͗L��.
// �c�[������͏����\��
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORKLAYER_H__

#ifdef NEURALNETWORKLAYER_EXPORTS
#define GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API __declspec(dllexport)
#else
#define GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.Utility.NeuralNetworkLayer.lib")
#endif
#endif


#include"../Layer/NeuralNetwork/ILayerDLLManager.h"
#include"../Layer/NeuralNetwork/ILayerDataManager.h"
#include"../Layer/Connect/ILayerConnectData.h"

#include<boost/filesystem.hpp>


namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	
	/** ���C���[DLL�Ǘ��N���X�̍쐬(CPU�p) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerCPU(const wchar_t i_libraryDirPath[]);
	/** ���C���[DLL�Ǘ��N���X�̍쐬(GPU�p) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::NeuralNetwork::ILayerDLLManager* CreateLayerDLLManagerGPU(const wchar_t i_libraryDirPath[]);

	//====================================
	// ���C���[�f�[�^���쐬
	//====================================
	/** �����j���[�����l�b�g���[�N.
		@param layerDLLManager	���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct	���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::Connect::ILayerConnectData* CreateNeuralNetwork(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct);

	/** �􍞂݃j���[�����l�b�g���[�N���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	filterSize			�t�B���^�T�C�Y.
		@param	outputChannelCount	�t�B���^�̌�.
		@param	stride				�t�B���^�̈ړ���.
		@param	paddingSize			�p�f�B���O�T�C�Y. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateConvolutionLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize);

	/** �S�����j���[�����l�b�g���[�N���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	neuronCount			�j���[������. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateFullyConnectLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct, U32 neuronCount);

	/** ���������C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	activationType		���������. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateActivationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct, const wchar_t activationType[]);

	/** �h���b�v�A�E�g���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	rate				�h���b�v�A�E�g��.(0.0�`1.0).(0.0���h���b�v�A�E�g�Ȃ�,1.0=�S���͖���) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateDropoutLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct, F32 rate);

	/** �v�[�����O���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	filterSize			�v�[�����O��.
		@param	stride				�t�B���^�ړ���. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreatePoolingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct, Vector3D<S32> filterSize, Vector3D<S32> stride);

	/** �o�b�`���K�����C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateBatchNormalizationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct);

	/** �L�敽�σv�[�����O���C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateGlobalAveragePoolingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct);

	/** GAN�ɂ�����Discriminator�̏o�̓��C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateActivationDiscriminatorLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct);


	/** �A�b�v�T���v�����O���C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	upScale				�g����.
		@param	paddingUseValue		�g�������̌����߂ɗאڂ���l���g�p����t���O. (true=UpConvolution, false=TransposeConvolution) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateUpSamplingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct, Vector3D<S32> upScale, bool paddingUseValue);

	/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	inputDataCount		���͂���郌�C���[�̌�. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeInputLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct lpInputDataStruct[], U32 inputDataCount);
	
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeInputLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const std::vector<IODataStruct>& lpInputDataStruct);

	template<typename... Rest>
	Layer::ILayerData* CreateMergeInputLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		std::vector<IODataStruct>& lpInputDataStruct, const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateMergeInputLayer(layerDLLManager, layerDataManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}
	template<typename... Rest>
	Layer::ILayerData* CreateMergeInputLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		std::vector<IODataStruct> lpInputDataStruct;
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateMergeInputLayer(layerDLLManager, layerDataManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}


	/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	inputDataCount		���͂���郌�C���[�̌�. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateResidualLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct lpInputDataStruct[], U32 inputDataCount);
	
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateResidualLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const std::vector<IODataStruct>& lpInputDataStruct);


	template<typename... Rest>
	Layer::ILayerData* CreateResidualLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		std::vector<IODataStruct>& lpInputDataStruct, const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateResidualLayer(layerDLLManager, layerDataManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}
	template<typename... Rest>
	Layer::ILayerData* CreateResidualLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& inputDataStruct_first, const Rest&... lpInputDataStruct_rest)
	{
		std::vector<IODataStruct> lpInputDataStruct;
		lpInputDataStruct.push_back(inputDataStruct_first);

		return CreateResidualLayer(layerDLLManager, layerDataManager, lpInputDataStruct, lpInputDataStruct_rest...);
	}


	/** ���C���[���l�b�g���[�N�̖����ɒǉ�����.GUID�͎������蓖��.���̓f�[�^�\���A�ŏIGUID���X�V����. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer);

	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer,
		const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount);

	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer,
		const std::vector<Gravisbell::GUID>& lpInputLayerGUID);

	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer,
		std::vector<Gravisbell::GUID>& lpInputLayerGUID,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		lpInputLayerGUID.push_back(lastLayerGUID_first);
	
		return AddLayerToNetworkLast(neuralNetwork, inputDataStruct, lastLayerGUID, pAddlayer, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}
	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::IODataStruct& inputDataStruct, Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		std::vector<Gravisbell::GUID> lpInputLayerGUID;
		lpInputLayerGUID.push_back(lastLayerGUID_first);

		return AddLayerToNetworkLast(neuralNetwork, inputDataStruct, lastLayerGUID, pAddlayer, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}


	/** �j���[�����l�b�g���[�N���o�C�i���t�@�C���ɕۑ����� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const wchar_t i_filePath[]);
	/** �j���[�����l�b�g���[�N���o�C�i���t�@�C������ǂݍ��ނ��� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const wchar_t i_filePath[]);


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif

