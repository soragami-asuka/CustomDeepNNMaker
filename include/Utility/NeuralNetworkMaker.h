//==================================
// �j���[�����l�b�g���[�N�̃��C���[�p�N���X
//==================================
#ifndef __GRAVISBELL_UTILITY_NEURALNETWORK_MAKER_H__
#define __GRAVISBELL_UTILITY_NEURALNETWORK_MAKER_H__

#include"NeuralNetworkLayer.h"

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	class INeuralNetworkMaker
	{
	public:
		/** �R���X�g���N�^ */
		INeuralNetworkMaker(){}
		/** �f�X�g���N�^ */
		virtual ~INeuralNetworkMaker(){}

	public:
		/** �쐬�����j���[�����l�b�g���[�N���擾���� */
		virtual Layer::Connect::ILayerConnectData* GetNeuralNetworkLayer()=0;

		/** �w�背�C���[�̏o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct(const Gravisbell::GUID& i_layerGUID) = 0;

	public:
		//==================================
		// ��{���C���[
		//==================================
		/** �􍞂݃j���[�����l�b�g���[�N���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_filterSize			�t�B���^�T�C�Y.
			@param	i_outputChannelCount	�t�B���^�̌�.
			@param	i_stride				�t�B���^�̈ړ���.
			@param	i_paddingSize			�p�f�B���O�T�C�Y. */
		virtual Gravisbell::GUID AddConvolutionLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize) = 0;

		/** �S�����j���[�����l�b�g���[�N���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_neuronCount			�j���[������. */
		virtual Gravisbell::GUID AddFullyConnectLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_neuronCount) = 0;

		/** ���������C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_activationType		���������. */
		virtual Gravisbell::GUID AddActivationLayer(const Gravisbell::GUID& i_inputLayerGUID, const wchar_t activationType[]) = 0;

		/** �h���b�v�A�E�g���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_rate				�h���b�v�A�E�g��.(0.0�`1.0).(0.0���h���b�v�A�E�g�Ȃ�,1.0=�S���͖���) */
		virtual Gravisbell::GUID AddDropoutLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_rate) = 0;

		/** �K�E�X�m�C�Y���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_average			�������闐���̕��ϒl
			@param	i_variance			�������闐���̕��U */
		virtual Gravisbell::GUID AddGaussianNoiseLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_average, F32 i_variance) = 0;

		/** �v�[�����O���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_filterSize			�v�[�����O��.
			@param	i_stride				�t�B���^�ړ���. */
		virtual Gravisbell::GUID AddPoolingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, Vector3D<S32> i_stride) = 0;

		/** �o�b�`���K�����C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		virtual Gravisbell::GUID AddBatchNormalizationLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** �o�b�`���K�����C���[(�`�����l����ʂȂ�)
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		virtual Gravisbell::GUID AddBatchNormalizationAllLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** �X�P�[�����K�����C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		virtual Gravisbell::GUID AddNormalizationScaleLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** �L�敽�σv�[�����O���C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		virtual Gravisbell::GUID AddGlobalAveragePoolingLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** GAN�ɂ�����Discriminator�̏o�̓��C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		virtual Gravisbell::GUID AddActivationDiscriminatorLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;

		/** �A�b�v�T���v�����O���C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_upScale				�g����.
			@param	i_paddingUseValue		�g�������̌����߂ɗאڂ���l���g�p����t���O. (true=UpConvolution, false=TransposeConvolution) */
		virtual Gravisbell::GUID AddUpSamplingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_upScale, bool i_paddingUseValue) = 0;

		/** �`�����l�����o���C���[. ���͂��ꂽ���C���[�̓���`�����l���𒊏o����. ����/�o�̓f�[�^�\����X,Y,Z�͓����T�C�Y.
			@param	i_inputLayerGUID	�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_startChannelNo	�J�n�`�����l���ԍ�.
			@param	i_channelCount		���o�`�����l����. */
		virtual Gravisbell::GUID AddChooseChannelLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_startChannelNo, U32 i_channelCount) = 0;

		/** �o�̓f�[�^�\���ϊ����C���[.
			@param	ch	CH��.
			@param	x	X��.
			@param	y	Y��.
			@param	z	Z��. */
		virtual Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 ch, U32 x, U32 y, U32 z) = 0;
		/** �o�̓f�[�^�\���ϊ����C���[.
			@param	outputDataStruct �o�̓f�[�^�\�� */
		virtual Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, const IODataStruct& outputDataStruct) = 0;

		/** X=0�Ń~���[������ */
		virtual Gravisbell::GUID AddReshapeMirrorXLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;
		/** X=0�ŕ��������� */
		virtual Gravisbell::GUID AddReshapeSquareCenterCrossLayer(const Gravisbell::GUID& i_inputLayerGUID) = 0;
		/** X=0�ŕ���������.
			���͐M������1�����z���(x-1)*(y-1)+1�ȏ�̗v�f�����K�v.
			@param	x	X��.
			@param	y	Y��. */
		virtual Gravisbell::GUID AddReshapeSquareZeroSideLeftTopLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 x, U32 y) = 0;


	protected:
		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		virtual Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeInputLayer(const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeInputLayer(&lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}
		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeInputLayer(std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeInputLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

	protected:
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�. */
		virtual Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�. */
		Gravisbell::GUID AddResidualLayer(const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddResidualLayer(&lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		template<typename... Rest>
		Gravisbell::GUID AddResidualLayer(std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddResidualLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

	public:
		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeInputLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddResidualLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	public:
		//==================================
		// ���ꃌ�C���[
		//==================================
		/** �j���[�����l�b�g���[�N��Convolution, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[]) = 0;

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[]) = 0;

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Noise, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBNA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_noiseVariance) = 0;

		/** �j���[�����l�b�g���[�N��BatchNormalization, Activation, Convolution���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[]) = 0;

		/** �j���[�����l�b�g���[�N��BatchNormalization, Activation. Fully-Connect���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BAF(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[]) = 0;

		/** �j���[�����l�b�g���[�N��Fully-Connect, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_FA(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[]) = 0;

		/** �j���[�����l�b�g���[�N��Fully-Connect, Activation, Dropout���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_FAD(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], F32 i_dropOutRate) = 0;

		/** �j���[�����l�b�g���[�N��BatchNormalization, Noise, Activation, Convolution���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BNAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance) = 0;

		/** �j���[�����l�b�g���[�N��Noise, BatchNormalization, Activation, Convolution���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_NBAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance) = 0;

		/** �j���[�����l�b�g���[�N��Convolution, Activation, DropOut���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate) = 0;

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Activation, DropOut���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate) = 0;

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����.(�m�C�Y�t��) */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNet(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance=0.0f) = 0;

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_front_layerCount, U32 i_back_layerCount, F32 i_noiseVariance=0.0f) = 0;

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����.
			�O��/�㔼�ɕ������Ɍ㔼���������ŏ���. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize_single(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance=0.0f) = 0;
	};

	/** �j���[�����l�b�g���[�N�쐬�N���X���擾���� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	INeuralNetworkMaker* CreateNeuralNetworkManaker(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount);

}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif