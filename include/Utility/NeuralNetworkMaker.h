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
		// ���C���[�ǉ�����
		//==================================
		virtual Gravisbell::GUID AddLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Gravisbell::Layer::ILayerData* i_pLayerData, bool onLayerFix=false) = 0;

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
		virtual Gravisbell::GUID AddConvolutionLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize,
			const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ���͊g���􍞂݃j���[�����l�b�g���[�N���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_filterSize			�t�B���^�T�C�Y.
			@param	i_outputChannelCount	�t�B���^�̌�.
			@param	i_dilation				���͂̊g����.
			@param	i_stride				�t�B���^�̈ړ���.
			@param	i_paddingSize			�p�f�B���O�T�C�Y. */
		virtual Gravisbell::GUID AddDilatedConvolutionLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_dilation, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize,
			const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;


		/** �S�����j���[�����l�b�g���[�N���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_neuronCount			�j���[������. */
		virtual Gravisbell::GUID AddFullyConnectLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_neuronCount, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** ���ȑg�D���}�b�v���C���[ */
		virtual Gravisbell::GUID AddSOMLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 dimensionCount=2, U32 resolutionCount=16, F32 initValueMin=0.0f, F32 initValueMax=1.0f, bool onLayerFix=false) = 0;

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

		/** XYZ���o���C���[. ���͂��ꂽ���C���[�̓���XYZ��Ԃ𒊏o����. ����/�o�̓f�[�^�\����CH�͓����T�C�Y.
			@param	startPosition	�J�nXYZ�ʒu.
			@param	boxSize			���oXYZ��. */
		virtual Gravisbell::GUID AddChooseBoxLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> startPosition, Vector3D<S32> boxSize) = 0;

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

		/** �M���̔z�񂩂�l�֕ϊ�.
			@param	outputMinValue	�o�͂̍ŏ��l
			@param	outputMaxValue	�o�͂̍ő�l
			@param	resolution		����\. �w�肵�Ȃ��ꍇ�͌����_�̏o�̓`�����l����������*/
		virtual Gravisbell::GUID AddSignalArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue) = 0;
		virtual Gravisbell::GUID AddSignalArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::U32 resolution) = 0;
		/** �m���̔z�񂩂�l�֕ϊ�.
			@param	outputMinValue	�o�͂̍ŏ��l
			@param	outputMaxValue	�o�͂̍ő�l
			@param	resolution		����\. �w�肵�Ȃ��ꍇ�͌����_�̏o�̓`�����l����������
			@param	variance		���U. ���͂ɑ΂��鋳�t�M�����쐬����ۂ̐��K���z�̕��U */
		virtual Gravisbell::GUID AddProbabilityArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance) = 0;
		virtual Gravisbell::GUID AddProbabilityArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance, Gravisbell::U32 resolution) = 0;
		/** �l����M���̔z��֕ϊ�.
			@param	outputMinValue	�o�͂̍ŏ��l
			@param	outputMaxValue	�o�͂̍ő�l
			@param	resolution		����\*/
		virtual Gravisbell::GUID AddValue2SignalArrayLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 inputMinValue, Gravisbell::F32 inputMaxValue, Gravisbell::U32 resolution) = 0;


	protected:
		//=============================
		// ���͌������C���[(�`�����l������)
		//=============================
		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		virtual Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeInputLayer(const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeInputLayer(&lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeInputLayer(std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeInputLayer(lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// ���͌������C���[(���Z)
		//=============================
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		virtual Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeAddLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeAddLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// ���͌������C���[(����)
		//=============================
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		virtual Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeAverageLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeAverageLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// ���͌������C���[(�ő�l)
		//=============================
		/** ���͌������C���[. ���͂��ꂽ���C���[�̍ő�l���Z�o����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		virtual Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[�̍ő�l���Z�o����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeMaxLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̍ő�l���Z�o����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeMaxLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}



	protected:
		//=============================
		// ���͌������C���[(��Z)
		//=============================
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		virtual Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;
		
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddMergeMultiplyLayer(i_layerMergeType, &lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}
		
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		template<typename... Rest>
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, std::vector<Gravisbell::GUID>& lpInputLayerGUID, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			lpInputLayerGUID.push_back(lastLayerGUID_first);
	
			return AddMergeMultiplyLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}


	protected:
		//=============================
		// ���͌������C���[(���Z)(����)
		//=============================
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�. */
		virtual Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount) = 0;

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�. */
		Gravisbell::GUID AddResidualLayer(const std::vector<Gravisbell::GUID>& lpInputLayerGUID)
		{
			return AddResidualLayer(&lpInputLayerGUID[0], (U32)lpInputLayerGUID.size());
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�. */
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

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeAddLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeAverageLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̍ő�l���Z�o����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeMaxLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
		}
		
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������.
			@param lastLayerGUID_first	
			@param lpLastLayerGUID_rest	
			*/
		template<typename... Rest>
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID& lastLayerGUID_first, const Rest&... lpLastLayerGUID_rest)
		{
			std::vector<Gravisbell::GUID> lpInputLayerGUID;
			lpInputLayerGUID.push_back(lastLayerGUID_first);

			return AddMergeMultiplyLayer(i_layerMergeType, lpInputLayerGUID, lpLastLayerGUID_rest...);
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
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Noise, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBNA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��BatchNormalization, Activation, Convolution���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��BatchNormalization, Activation. Fully-Connect���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BAF(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Fully-Connect, Activation���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_FA(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Fully-Connect, Activation, Dropout���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_FAD(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��BatchNormalization, Noise, Activation, Convolution���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_BNAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Noise, BatchNormalization, Activation, Convolution���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_NBAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Convolution, Activation, DropOut���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Activation, DropOut���s�����C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_CBAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����.(�m�C�Y�t��) */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNet(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance=0.0f, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_front_layerCount, U32 i_back_layerCount, F32 i_noiseVariance=0.0f, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����.
			�O��/�㔼�ɕ������Ɍ㔼���������ŏ���. */
		virtual Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize_single(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance=0.0f, const wchar_t i_szInitializerID[] = L"glorot_uniform") = 0;
	};

	/** �j���[�����l�b�g���[�N�쐬�N���X���擾���� */
	GRAVISBELL_UTILITY_NEURALNETWORKLAYER_API
	INeuralNetworkMaker* CreateNeuralNetworkManaker(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount);

}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell

#endif