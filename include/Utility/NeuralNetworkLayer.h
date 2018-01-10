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
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

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
		U32 inputChannelCount, Vector3D<S32> filterSize, U32 outputChannelCount, Vector3D<S32> stride, Vector3D<S32> paddingSize,
		const wchar_t i_szInitializerID[] = L"glorot_uniform");

	/** �S�����j���[�����l�b�g���[�N���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	neuronCount			�j���[������. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateFullyConnectLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputBufferCount, U32 neuronCount,
		const wchar_t i_szInitializerID[] = L"glorot_uniform");

	/** ���������C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	activationType		���������. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateActivationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const wchar_t activationType[]);

	/** �h���b�v�A�E�g���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	rate				�h���b�v�A�E�g��.(0.0�`1.0).(0.0���h���b�v�A�E�g�Ȃ�,1.0=�S���͖���) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateDropoutLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		F32 rate);

	/** �K�E�X�m�C�Y���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	average				�������闐���̕��ϒl
		@param	variance			�������闐���̕��U */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateGaussianNoiseLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, F32 average, F32 variance);


	/** �v�[�����O���C���[.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	filterSize			�v�[�����O��.
		@param	stride				�t�B���^�ړ���. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreatePoolingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Vector3D<S32> filterSize, Vector3D<S32> stride);

	/** �o�b�`���K�����C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateBatchNormalizationLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 inputChannelCount);

	/** �o�b�`���K�����C���[(�`�����l����ʂȂ�)
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateBatchNormalizationAllLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** �X�P�[�����K�����C���[ */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateNormalizationScaleLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** �L�敽�σv�[�����O���C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateGlobalAveragePoolingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** GAN�ɂ�����Discriminator�̏o�̓��C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateActivationDiscriminatorLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);


	/** �A�b�v�T���v�����O���C���[
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	upScale				�g����.
		@param	paddingUseValue		�g�������̌����߂ɗאڂ���l���g�p����t���O. (true=UpConvolution, false=TransposeConvolution) */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateUpSamplingLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Vector3D<S32> upScale, bool paddingUseValue);

	/** �`�����l�����o���C���[. ���͂��ꂽ���C���[�̓���`�����l���𒊏o����. ����/�o�̓f�[�^�\����X,Y,Z�͓����T�C�Y.
		@param	startChannelNo	�J�n�`�����l���ԍ�.
		@param	channelCount	���o�`�����l����. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateChooseChannelLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 startChannelNo, U32 channelCount);

	/** �o�̓f�[�^�\���ϊ����C���[.
		@param	ch	CH��.
		@param	x	X��.
		@param	y	Y��.
		@param	z	Z��. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		U32 ch, U32 x, U32 y, U32 z);
	/** �o�̓f�[�^�\���ϊ����C���[.
		@param	outputDataStruct �o�̓f�[�^�\�� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		const IODataStruct& outputDataStruct);

	/** X=0�𒆐S�Ƀ~���[������*/
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeMirrorXLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** X=0�𒆐S�ɕ���������. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeSquareCenterCrossLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);

	/** X=0�𒆐S�ɕ���������. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateReshapeSquareZeroSideLeftTopLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager,
		Gravisbell::U32 x, Gravisbell::U32 y);


	/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	inputDataCount		���͂���郌�C���[�̌�. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateMergeInputLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);


	/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�.
		@param	layerDLLManager		���C���[DLL�Ǘ��N���X.
		@param	inputDataStruct		���̓f�[�^�\��.
		@param	inputDataCount		���͂���郌�C���[�̌�. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	Layer::ILayerData* CreateResidualLayer(
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager);



	/** ���C���[���l�b�g���[�N�̖����ɒǉ�����.GUID�͎������蓖��.���̓f�[�^�\���A�ŏIGUID���X�V����. */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix);

	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer, bool onLayerFix,
		const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount);

	
	inline Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddLayer, bool onLayerFix,
		const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
	{
		return AddLayerToNetworkLast(neuralNetwork, lastLayerGUID, pAddLayer, onLayerFix, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
	}

	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix,
		std::vector<Gravisbell::GUID>& lpInputLayerGUID,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		lpInputLayerGUID.push_back(lastLayerGUID_first);

		return AddLayerToNetworkLast(neuralNetwork, lastLayerGUID, pAddlayer, onLayerFix, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}
	template<typename... Rest>
	Gravisbell::ErrorCode AddLayerToNetworkLast(
		Layer::Connect::ILayerConnectData& neuralNetwork,
		Gravisbell::GUID& lastLayerGUID, Layer::ILayerData* pAddlayer, bool onLayerFix,
		const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
	{
		std::vector<Gravisbell::GUID> lpInputLayerGUID;
		lpInputLayerGUID.push_back(lastLayerGUID_first);

		return AddLayerToNetworkLast(neuralNetwork, lastLayerGUID, pAddlayer, onLayerFix, lpInputLayerGUID, lpLastLayerGUID_rest...);
	}


	/** �j���[�����l�b�g���[�N���o�C�i���t�@�C���ɕۑ����� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode WriteNetworkToBinaryFile(const Layer::ILayerData& neuralNetwork, const wchar_t i_filePath[]);
	/** �j���[�����l�b�g���[�N���o�C�i���t�@�C������ǂݍ��ނ��� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API Gravisbell::ErrorCode ReadNetworkFromBinaryFile(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::ILayerData** ppNeuralNetwork, const wchar_t i_filePath[]);


}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif

