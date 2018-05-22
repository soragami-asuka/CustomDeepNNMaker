//==================================
// �j���[�����l�b�g���[�N�̃��C���[�Ǘ��p��Utiltiy
// ���C�u�����Ƃ��Ďg���Ԃ͗L��.
// �c�[������͏����\��
//==================================
#include"stdafx.h"

#include"Utility/NeuralNetworkMaker.h"

#include<vector>

namespace Gravisbell {
namespace Utility {
namespace NeuralNetworkLayer {

	class NeuralNetworkMaker : public INeuralNetworkMaker
	{
	private:
		const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager;	/**< DLL�Ǘ��N���X */
		Layer::NeuralNetwork::ILayerDataManager& layerDataManager;	/**< ���C���[�Ǘ��N���X */

		Layer::Connect::ILayerConnectData* pLayerConnectData;		/**< ���C���[�ڑ���� */
		bool onGetConnectData;

		std::vector<IODataStruct> lpInputDataStruct;

	public:
		/** �R���X�g���N�^ */
		NeuralNetworkMaker(const Layer::NeuralNetwork::ILayerDLLManager& i_layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& i_layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount)
			:	layerDLLManager		(i_layerDLLManager)
			,	layerDataManager	(i_layerDataManager)
			,	pLayerConnectData	(CreateNeuralNetwork(layerDLLManager,layerDataManager))
			,	onGetConnectData	(false)
		{
			for(U32 i=0; i<i_inputCount; i++)
				this->lpInputDataStruct.push_back(i_lpInputDataStruct[i]);
		}
		/** �f�X�g���N�^ */
		virtual ~NeuralNetworkMaker()
		{
			if(!onGetConnectData)
				delete pLayerConnectData;
		}

		/** �쐬�����j���[�����l�b�g���[�N���擾���� */
		Layer::Connect::ILayerConnectData* GetNeuralNetworkLayer()
		{
			this->onGetConnectData = true;

			return this->pLayerConnectData;
		}

		/** �w�背�C���[�̏o�̓f�[�^�\�����擾���� */
		IODataStruct GetOutputDataStruct(const Gravisbell::GUID& i_layerGUID)
		{
			return this->pLayerConnectData->GetOutputDataStruct(i_layerGUID, &this->lpInputDataStruct[0], (U32)this->lpInputDataStruct.size());
		}

	public:
		//==================================
		// ���C���[�ǉ�����
		//==================================
		Gravisbell::GUID AddLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			Gravisbell::Layer::ILayerData* i_pLayerData, bool onLayerFix=false)
		{
			Gravisbell::GUID layerGUID = i_inputLayerGUID;

			if(i_pLayerData == NULL)
				return Gravisbell::GUID();

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				i_pLayerData,
				onLayerFix);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return Gravisbell::GUID();

			return layerGUID;
		}

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
		Gravisbell::GUID AddConvolutionLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateConvolutionLayer(layerDLLManager, layerDataManager, this->GetOutputDataStruct(i_inputLayerGUID).ch, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID),
				false);
		}
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
			const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateDilatedConvolutionLayer(layerDLLManager, layerDataManager, this->GetOutputDataStruct(i_inputLayerGUID).ch, i_filterSize, i_outputChannelCount, i_dilation, i_stride, i_paddingSize, i_szInitializerID),
				false);
		}

		/** �S�����j���[�����l�b�g���[�N���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_neuronCount			�j���[������. */
		Gravisbell::GUID AddFullyConnectLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_neuronCount, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateFullyConnectLayer(layerDLLManager, layerDataManager, this->GetOutputDataStruct(i_inputLayerGUID).GetDataCount(), i_neuronCount, i_szInitializerID),
				false);
		}

		/** ���ȑg�D���}�b�v���C���[ */
		Gravisbell::GUID AddSOMLayer(
			const Gravisbell::GUID& i_inputLayerGUID,
			U32 dimensionCount=2, U32 resolutionCount=16,
			F32 initValueMin=0.0f, F32 initValueMax=1.0f,
			bool onLayerFix=false)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateSOMLayer(layerDLLManager, layerDataManager, this->GetOutputDataStruct(i_inputLayerGUID).GetDataCount(), dimensionCount, resolutionCount, initValueMin, initValueMax),
				onLayerFix);
		}

		/** ���������C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_activationType		���������. */
		Gravisbell::GUID AddActivationLayer(const Gravisbell::GUID& i_inputLayerGUID, const wchar_t activationType[])
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateActivationLayer(layerDLLManager, layerDataManager, activationType),
				false);
		}

		/** �h���b�v�A�E�g���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_rate				�h���b�v�A�E�g��.(0.0�`1.0).(0.0���h���b�v�A�E�g�Ȃ�,1.0=�S���͖���) */
		Gravisbell::GUID AddDropoutLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_rate)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateDropoutLayer(layerDLLManager, layerDataManager, i_rate),
				false);
		}

		/** �K�E�X�m�C�Y���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_average			�������闐���̕��ϒl
			@param	i_variance			�������闐���̕��U */
		Gravisbell::GUID AddGaussianNoiseLayer(const Gravisbell::GUID& i_inputLayerGUID, F32 i_average, F32 i_variance)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateGaussianNoiseLayer(layerDLLManager, layerDataManager, i_average, i_variance),
				false);
		}

		/** �v�[�����O���C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_filterSize			�v�[�����O��.
			@param	i_stride				�t�B���^�ړ���. */
		Gravisbell::GUID AddPoolingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, Vector3D<S32> i_stride)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreatePoolingLayer(layerDLLManager, layerDataManager, i_filterSize, i_stride),
				false);
		}

		/** �o�b�`���K�����C���[.
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		Gravisbell::GUID AddBatchNormalizationLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateBatchNormalizationLayer(layerDLLManager, layerDataManager, this->GetOutputDataStruct(i_inputLayerGUID).ch),
				false);
		}

		/** �o�b�`���K�����C���[(�`�����l����ʂȂ�)
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		Gravisbell::GUID AddBatchNormalizationAllLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateBatchNormalizationAllLayer(layerDLLManager, layerDataManager),
				false);
		}

		/** �X�P�[�����K�����C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		Gravisbell::GUID AddNormalizationScaleLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateNormalizationScaleLayer(layerDLLManager, layerDataManager),
				false);
		}

		/** �L�敽�σv�[�����O���C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		Gravisbell::GUID AddGlobalAveragePoolingLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateGlobalAveragePoolingLayer(layerDLLManager, layerDataManager),
				false);
		}

		/** GAN�ɂ�����Discriminator�̏o�̓��C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID. */
		Gravisbell::GUID AddActivationDiscriminatorLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateActivationDiscriminatorLayer(layerDLLManager, layerDataManager),
				false);
		}

		/** �A�b�v�T���v�����O���C���[
			@param	i_inputLayerGUID		�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_upScale				�g����.
			@param	i_paddingUseValue		�g�������̌����߂ɗאڂ���l���g�p����t���O. (true=UpConvolution, false=TransposeConvolution) */
		Gravisbell::GUID AddUpSamplingLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> i_upScale, bool i_paddingUseValue)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateUpSamplingLayer(layerDLLManager, layerDataManager, i_upScale, i_paddingUseValue),
				false);
		}

		/** �`�����l�����o���C���[. ���͂��ꂽ���C���[�̓���`�����l���𒊏o����. ����/�o�̓f�[�^�\����X,Y,Z�͓����T�C�Y.
			@param	i_inputLayerGUID	�ǉ����C���[�̓��͐惌�C���[��GUID.
			@param	i_startChannelNo	�J�n�`�����l���ԍ�.
			@param	i_channelCount		���o�`�����l����. */
		Gravisbell::GUID AddChooseChannelLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 i_startChannelNo, U32 i_channelCount)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateChooseChannelLayer(layerDLLManager, layerDataManager, i_startChannelNo, i_channelCount),
				false);
		}

		/** XYZ���o���C���[. ���͂��ꂽ���C���[�̓���XYZ��Ԃ𒊏o����. ����/�o�̓f�[�^�\����CH�͓����T�C�Y.
			@param	startPosition	�J�nXYZ�ʒu.
			@param	boxSize			���oXYZ��. */
		Gravisbell::GUID AddChooseBoxLayer(const Gravisbell::GUID& i_inputLayerGUID, Vector3D<S32> startPosition, Vector3D<S32> boxSize)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateChooseBoxLayer(layerDLLManager, layerDataManager, startPosition, boxSize),
				false);
		}


		/** �o�̓f�[�^�\���ϊ����C���[.
			@param	ch	CH��.
			@param	x	X��.
			@param	y	Y��.
			@param	z	Z��. */
		Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 ch, U32 x, U32 y, U32 z)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateReshapeLayer(layerDLLManager, layerDataManager, ch, x, y, z),
				false);
		}
		/** �o�̓f�[�^�\���ϊ����C���[.
			@param	outputDataStruct �o�̓f�[�^�\�� */
		Gravisbell::GUID AddReshapeLayer(const Gravisbell::GUID& i_inputLayerGUID, const IODataStruct& outputDataStruct)
		{
			return AddReshapeLayer(i_inputLayerGUID, outputDataStruct.ch, outputDataStruct.x, outputDataStruct.y, outputDataStruct.z);
		}

		/** X=0�Ń~���[������ */
		Gravisbell::GUID AddReshapeMirrorXLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateReshapeMirrorXLayer(layerDLLManager, layerDataManager),
				false);
		}
		/** X=0�ŕ��������� */
		Gravisbell::GUID AddReshapeSquareCenterCrossLayer(const Gravisbell::GUID& i_inputLayerGUID)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateReshapeSquareCenterCrossLayer(layerDLLManager, layerDataManager),
				false);
		}
		/** X=0�ŕ���������.
			���͐M������1�����z���(x-1)*(y-1)+1�ȏ�̗v�f�����K�v.
			@param	x	X��.
			@param	y	Y��. */
		Gravisbell::GUID AddReshapeSquareZeroSideLeftTopLayer(const Gravisbell::GUID& i_inputLayerGUID, U32 x, U32 y)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateReshapeSquareZeroSideLeftTopLayer(layerDLLManager, layerDataManager, x, y),
				false);
		}

		/** �M���̔z�񂩂�l�֕ϊ� */
		Gravisbell::GUID AddSignalArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateSignalArray2ValueLayer(layerDLLManager, layerDataManager, outputMinValue, outputMaxValue, this->GetOutputDataStruct(i_inputLayerGUID).ch),
				false);
		}
		Gravisbell::GUID AddSignalArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::U32 resolution)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateSignalArray2ValueLayer(layerDLLManager, layerDataManager, outputMinValue, outputMaxValue, resolution),
				false);
		}
		/** �m���̔z�񂩂�l�֕ϊ�.
			@param	outputMinValue	�o�͂̍ŏ��l
			@param	outputMaxValue	�o�͂̍ő�l
			@param	resolution		����\. �w�肵�Ȃ��ꍇ�͌����_�̏o�̓`�����l����������
			@param	variance		���U. ���͂ɑ΂��鋳�t�M�����쐬����ۂ̐��K���z�̕��U */
		Gravisbell::GUID AddProbabilityArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateProbabilityArray2ValueLayer(layerDLLManager, layerDataManager, outputMinValue, outputMaxValue, variance, this->GetOutputDataStruct(i_inputLayerGUID).ch),
				false);
		}
		Gravisbell::GUID AddProbabilityArray2ValueLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 outputMinValue, Gravisbell::F32 outputMaxValue, Gravisbell::F32 variance, Gravisbell::U32 resolution)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateProbabilityArray2ValueLayer(layerDLLManager, layerDataManager, outputMinValue, outputMaxValue, variance, resolution),
				false);
		}
		/** �l����M���̔z��֕ϊ�.
			@param	outputMinValue	�o�͂̍ŏ��l
			@param	outputMaxValue	�o�͂̍ő�l
			@param	resolution		����\*/
		Gravisbell::GUID AddValue2SignalArrayLayer(const Gravisbell::GUID& i_inputLayerGUID, Gravisbell::F32 inputMinValue, Gravisbell::F32 inputMaxValue, Gravisbell::U32 resolution)
		{
			return this->AddLayer(
				i_inputLayerGUID,
				CreateValue2SignalArrayLayer(layerDLLManager, layerDataManager, inputMinValue, inputMaxValue, resolution),
				false);
		}

	public:
		/** ���͌������C���[. ���͂��ꂽ���C���[��CH����������. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeInputLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateMergeInputLayer(layerDLLManager, layerDataManager), false,
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeAddLayer(LayerMergeType i_layerMergeType, Gravisbell::F32 i_scale, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateMergeAddLayer(layerDLLManager, layerDataManager, i_layerMergeType, i_scale), false,
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeAverageLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateMergeAverageLayer(layerDLLManager, layerDataManager, i_layerMergeType), false,
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}
		
		/** ���͌������C���[. ���͂��ꂽ���C���[�̍ő�l���Z�o����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeMaxLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateMergeMaxLayer(layerDLLManager, layerDataManager, i_layerMergeType), false,
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}
		
		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l����Z����. ���̓f�[�^�\����X,Y,Z�œ����T�C�Y�ł���K�v������. */
		Gravisbell::GUID AddMergeMultiplyLayer(LayerMergeType i_layerMergeType, const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateMergeMultiplyLayer(layerDLLManager, layerDataManager, i_layerMergeType), false,
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}

		/** ���͌������C���[. ���͂��ꂽ���C���[�̒l�����Z����. �o�͂���郌�C���[�̃T�C�Y�͑S�T�C�Y�̂����̍ő�l�ɂȂ�. */
		Gravisbell::GUID AddResidualLayer(const Gravisbell::GUID lpInputLayerGUID[], U32 inputLayerCount)
		{
			Gravisbell::GUID layerGUID;

			Gravisbell::ErrorCode err = AddLayerToNetworkLast(
				*this->pLayerConnectData,
				layerGUID,
				CreateResidualLayer(layerDLLManager, layerDataManager), false,
				lpInputLayerGUID, inputLayerCount);

			return layerGUID;
		}

		
	public:
		//==================================
		// ���ꃌ�C���[
		//==================================
		/** �j���[�����l�b�g���[�N��Convolution, Activation���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_CA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Activation���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_CBA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Noise, Activation���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_CBNA(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// �m�C�Y
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��BatchNormalization, Activation, Convolution���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_BAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��BatchNormalization, Activation. Fully-Connect���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_BAF(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// �S����
			layerGUID = this->AddFullyConnectLayer(layerGUID, i_outputChannelCount, i_szInitializerID);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��Fully-Connect, Activation���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_FA(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			return AddNeuralNetworkLayer_FAD(i_inputLayerGUID, i_outputChannelCount, i_activationType, 0.0f, i_szInitializerID);
		}

		/** �j���[�����l�b�g���[�N��Fully-Connect, Activation, Dropout���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_FAD(const GUID& i_inputLayerGUID, U32 i_outputChannelCount, const wchar_t i_activationType[], F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// �S����
			layerGUID = this->AddFullyConnectLayer(layerGUID, i_outputChannelCount, i_szInitializerID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// �h���b�v�A�E�g
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��BatchNormalization, Noise, Activation, Convolution���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_BNAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// �m�C�Y
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��Noise, BatchNormalization, Activation, Convolution���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_NBAC(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// �m�C�Y
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��Convolution, Activation, DropOut���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_CAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// �h���b�v�A�E�g
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��Convolution, BatchNormalization, Activation, DropOut���s�����C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_CBAD(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, Vector3D<S32> i_stride, Vector3D<S32> i_paddingSize, const wchar_t i_activationType[], Gravisbell::F32 i_dropOutRate, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID layerGUID = i_inputLayerGUID;

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, i_stride, i_paddingSize, i_szInitializerID);

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, i_activationType);

			// �h���b�v�A�E�g
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			return layerGUID;
		}

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����.(�m�C�Y�t��) */
		Gravisbell::GUID AddNeuralNetworkLayer_ResNet(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID bypassLayerGUID = i_inputLayerGUID;
			GUID layerGUID = i_inputLayerGUID;

			U32 outputChannel = this->GetOutputDataStruct(i_inputLayerGUID).ch;

			for(U32 layerNum=0; layerNum<i_layerCount-1; layerNum++)
			{
				// 2�w��
				layerGUID = this->AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, outputChannel, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU", i_szInitializerID);
			}

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// �m�C�Y
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, L"ReLU");

			// �h���b�v�A�E�g
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, outputChannel, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), i_szInitializerID );

			// Residual
			layerGUID = INeuralNetworkMaker::AddResidualLayer(layerGUID, bypassLayerGUID);


			return S_OK;
		}

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_front_layerCount, U32 i_back_layerCount, F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			GUID bypassLayerGUID = i_inputLayerGUID;
			GUID layerGUID = i_inputLayerGUID;

			U32 inputChannelCount = this->GetOutputDataStruct(i_inputLayerGUID).ch;

			// �O��
			for(S32 layerNum=0; layerNum<(S32)i_front_layerCount-1; layerNum++)
			{
				layerGUID = this->AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, inputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU", i_szInitializerID);
			}

			// CH����ύX
			layerGUID = this->AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, i_outputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU", i_szInitializerID);

			// �㔼
			for(S32 layerNum=0; layerNum<(S32)i_back_layerCount-1; layerNum++)
			{
				// 2�w��
				layerGUID = AddNeuralNetworkLayer_BAC(layerGUID, i_filterSize, i_outputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), L"ReLU", i_szInitializerID);
			}

			// �o�b�`���K��
			layerGUID = this->AddBatchNormalizationLayer(layerGUID);

			// �m�C�Y
			if(i_noiseVariance > 0.0f)
				layerGUID = this->AddGaussianNoiseLayer(layerGUID, 0.0f, i_noiseVariance);

			// ������
			layerGUID = this->AddActivationLayer(layerGUID, L"ReLU");

			// �h���b�v�A�E�g
			if(i_dropOutRate > 0.0f)
				layerGUID = this->AddDropoutLayer(layerGUID, i_dropOutRate);

			// ��ݍ���
			layerGUID = this->AddConvolutionLayer(layerGUID, i_filterSize, i_outputChannelCount, Vector3D<S32>(1,1,1), Vector3D<S32>(i_filterSize.x/2,i_filterSize.y/2,i_filterSize.z/2), i_szInitializerID );

			// Residual
			layerGUID = INeuralNetworkMaker::AddResidualLayer(layerGUID, bypassLayerGUID);


			return S_OK;
		}

		/** �j���[�����l�b�g���[�N��ResNet���C���[��ǉ�����. */
		Gravisbell::GUID AddNeuralNetworkLayer_ResNetResize_single(const GUID& i_inputLayerGUID, Vector3D<S32> i_filterSize, U32 i_outputChannelCount, F32 i_dropOutRate, U32 i_layerCount, F32 i_noiseVariance, const wchar_t i_szInitializerID[] = L"glorot_uniform")
		{
			return this->AddNeuralNetworkLayer_ResNetResize(i_inputLayerGUID, i_filterSize, i_outputChannelCount, i_dropOutRate, 1, max(0, (Gravisbell::S32)i_layerCount-1), i_noiseVariance, i_szInitializerID);
		}
	};

	/** �j���[�����l�b�g���[�N�쐬�N���X���擾���� */
	INeuralNetworkMaker* CreateNeuralNetworkManaker(const Layer::NeuralNetwork::ILayerDLLManager& layerDLLManager, Layer::NeuralNetwork::ILayerDataManager& layerDataManager, const IODataStruct i_lpInputDataStruct[], U32 i_inputCount)
	{
		return new NeuralNetworkMaker(layerDLLManager, layerDataManager, i_lpInputDataStruct, i_inputCount);
	}

}	// NeuralNetworkLayer
}	// Utility
}	// Gravisbell
