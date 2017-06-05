//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"UpConvolution_DATA.hpp"
#include"UpConvolution_FUNC.hpp"
#include"UpConvolution_Base.h"

#include"UpConvolution_GPU.cuh"
#include"UpConvolution_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	UpConvolution_GPU::UpConvolution_GPU(Gravisbell::GUID guid, UpConvolution_LayerData_GPU& i_layerData)
		:	UpConvolution_Base	(guid)
		,	layerData			(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount	(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount			(0)		/**< �j���[������ */
		,	outputBufferCount	(0)		/**< �o�̓o�b�t�@�� */
		,	cudnnHandle			(NULL)
		,	inputTensorDesc		(NULL)
		,	outputTensorDesc	(NULL)
		,	biasTensorDesc		(NULL)
		,	filterDesc			(NULL)
		,	convDesc			(NULL)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateTensorDescriptor(&biasTensorDesc);
		cudnnCreateFilterDescriptor(&filterDesc);
		cudnnCreateConvolutionDescriptor(&convDesc);
	}
	/** �f�X�g���N�^ */
	UpConvolution_GPU::~UpConvolution_GPU()
	{
		if(convDesc)			cudnnDestroyConvolutionDescriptor(convDesc);
		if(filterDesc)			cudnnDestroyFilterDescriptor(filterDesc);
		if(biasTensorDesc)		cudnnDestroyTensorDescriptor(biasTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 UpConvolution_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode UpConvolution_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	UpConvolution_LayerData_Base& UpConvolution_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const UpConvolution_LayerData_Base& UpConvolution_GPU::GetLayerData()const
	{
		return this->layerData;
	}



	//================================
	// ���Z����
	//================================
	/** ���Z�O���������s����.(�w�K�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
	ErrorCode UpConvolution_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͍����o�b�t�@���쐬
		this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);

		// �j���[����/�o�C�A�X�̌덷���ꎞ�ۑ�����o�b�t�@���쐬
		this->lpDBias.resize(this->layerData.lpBias_d.size());
		this->lppDNeuron.resize(this->layerData.lppNeuron_d.size());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpConvolution_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->neuronCount = this->layerData.layerStructure.Output_Channel;
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		cudnnStatus_t err_cudnn;

		// �������𒲂ׂ�
		S32 dataDim = 1 + 1 + 0;	// �o�b�` + �`�����l�� + ����0
		std::vector<S32> dimInput;			// ���̓f�[�^�\��
		std::vector<S32> dimInputStride;	// ���̓f�[�^�̊e�������Ƃ̃f�[�^��
		std::vector<S32> dimBias;
		std::vector<S32> dimBiasStride;
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		S32 filterDim = 0;			// �t�B���^������	���̓`�����l�� + �o�̓`�����l�� + ����
		std::vector<S32> dimFilter;
		S32 convDim = 0;			// ��ݍ��ݎ�����	����
		std::vector<S32> dimStride;
		std::vector<S32> dimDilation;
		std::vector<S32> dimPadding;
		if(this->layerData.inputDataStruct.z > 1)
		{
			dataDim = 1 + 1 + 3;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.z * this->layerData.layerStructure.UpScale.x;
			dimInput[3] = this->layerData.inputDataStruct.y * this->layerData.layerStructure.UpScale.y;
			dimInput[4] = this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.z;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->layerData.GetOutputDataStruct().ch;
			dimBias[2] = 1;
			dimBias[3] = 1;
			dimBias[4] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2] * dimBias[3] * dimBias[4];
			dimBiasStride[1] = dimBias[2] * dimBias[3] * dimBias[4];
			dimBiasStride[2] = dimBias[3] * dimBias[4];
			dimBiasStride[3] = dimBias[4];
			dimBiasStride[4] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 3;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.GetOutputDataStruct().ch;
			dimFilter[1] = this->layerData.inputDataStruct.ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.z;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[4] = this->layerData.layerStructure.FilterSize.x;

			convDim = 3;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.z;
			dimPadding[1] = this->layerData.layerStructure.Padding.y;
			dimPadding[2] = this->layerData.layerStructure.Padding.x;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;
			dimDilation[1] = 1;
			dimDilation[2] = 1;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.z;
			dimStride[1] = this->layerData.layerStructure.Stride.y;
			dimStride[2] = this->layerData.layerStructure.Stride.x;
		}
		else if(this->layerData.inputDataStruct.y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.y * this->layerData.layerStructure.UpScale.y;
			dimInput[3] = this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->layerData.GetOutputDataStruct().ch;
			dimBias[2] = 1;
			dimBias[3] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2] * dimBias[3];
			dimBiasStride[1] = dimBias[2] * dimBias[3];
			dimBiasStride[2] = dimBias[3];
			dimBiasStride[3] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 2;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.GetOutputDataStruct().ch;
			dimFilter[1] = this->layerData.inputDataStruct.ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.x;

			convDim = 2;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.y;
			dimPadding[1] = this->layerData.layerStructure.Padding.x;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;
			dimDilation[1] = 1;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.y;
			dimStride[1] = this->layerData.layerStructure.Stride.x;
		}
		else if(this->layerData.inputDataStruct.x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->layerData.GetOutputDataStruct().ch;
			dimBias[2] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2];
			dimBiasStride[1] = dimBias[2];
			dimBiasStride[2] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 1;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.GetOutputDataStruct().ch;
			dimFilter[1] = this->layerData.inputDataStruct.ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.x;

			convDim = 1;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.x;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.x;
		}
		else
		{
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}

		// CUDNN�̓��̓f�[�^�\����ݒ�
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->inputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimInputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// �t�B���^�T�C�Y��ݒ�
		err_cudnn = cudnnSetFilterNdDescriptor(
			this->filterDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NCHW,
			filterDim,
			&dimFilter[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// ��ݍ��ݏ����ݒ�
		err_cudnn = cudnnSetConvolutionNdDescriptor(
			this->convDesc,
			convDim,
			&dimPadding[0],
			&dimStride[0],
			&dimDilation[0],
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �o�̓f�[�^�\�����擾
        err_cudnn = cudnnGetConvolutionNdForwardOutputDim(
			this->convDesc,
			this->inputTensorDesc,
			this->filterDesc,
			dataDim,
			&dimOutput[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNN�̏o�̓f�[�^�\����Gravisbell�̏o�̓f�[�^�\������v���邱�Ƃ��m�F
		Gravisbell::Vector3D<S32> outputVector;
		S32 outputBatchSize = dimOutput[0];
		S32 outputCh = dimOutput[1];
		if(dataDim == 5)
		{
			outputVector.z = dimOutput[2];
			outputVector.y = dimOutput[3];
			outputVector.x = dimOutput[4];
		}
		else if(dataDim == 4)
		{
			outputVector.z = 1;
			outputVector.y = dimOutput[2];
			outputVector.x = dimOutput[3];
		}
		else if(dataDim == 3)
		{
			outputVector.z = 1;
			outputVector.y = 1;
			outputVector.x = dimOutput[2];
		}
		if(outputBatchSize != this->batchSize)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputCh != this->GetOutputDataStruct().ch)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.z != this->GetOutputDataStruct().z)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.y != this->GetOutputDataStruct().y)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.x != this->GetOutputDataStruct().x)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNN�̏o�̓f�[�^�\����ݒ�
		dimOutputStride.resize(dataDim);
		for(S32 i=0; i<dataDim; i++)
		{
			dimOutputStride[i] = 1;
			for(S32 j=i+1; j<dataDim; j++)
				dimOutputStride[i] *= dimOutput[j];
		}
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// �ő��̃A���S���Y������������(�O���`�d)
		err_cudnn = cudnnGetConvolutionForwardAlgorithm(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,	// �������̎g�p�ʖ������ōő��̃A���S���Y���𒲂ׂ�
			0,										// �g�p�\�ȃ������̏��
			&this->useForwardAlgorithm
			);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �K�v�ȃ������ʂ𒲂ׂ�(�O���`�d)
		size_t workSpaceSizeByte_forward;
		err_cudnn = cudnnGetConvolutionForwardWorkspaceSize(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->useForwardAlgorithm,
			&workSpaceSizeByte_forward);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �ő��̃A���S���Y������������(����`�d-�f�[�^)
		err_cudnn = cudnnGetConvolutionBackwardDataAlgorithm(
			this->cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			cudnnConvolutionBwdDataPreference_t::CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,	// �������̎g�p�ʖ������ōő��̃A���S���Y���𒲂ׂ�
			0,																				// �g�p�\�ȃ������̏��
			&this->useBackwardDataAlgorithm);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �K�v�ȃ������ʂ𒲂ׂ�(����`�d-�f�[�^)
		size_t workSpaceSizeByte_backwardData;
		err_cudnn = cudnnGetConvolutionBackwardDataWorkspaceSize(
			this->cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->useBackwardDataAlgorithm,
			&workSpaceSizeByte_backwardData);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �ő��̃A���S���Y������������(����`�d-�f�[�^)
		err_cudnn = cudnnGetConvolutionBackwardFilterAlgorithm(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			cudnnConvolutionBwdFilterPreference_t::CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,	// �������̎g�p�ʖ������ōő��̃A���S���Y���𒲂ׂ�
			0,																					// �g�p�\�ȃ������̏��
			&this->useBackwardFilterAlgorithm);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// �K�v�ȃ������ʂ𒲂ׂ�(����`�d-�f�[�^)
		size_t workSpaceSizeByte_backwardFilter;
		err_cudnn = cudnnGetConvolutionBackwardFilterWorkspaceSize(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->useBackwardFilterAlgorithm,
			&workSpaceSizeByte_backwardFilter);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �����p�o�b�t�@�̊m��
		this->workSpace.resize(max(workSpaceSizeByte_forward, max(workSpaceSizeByte_backwardData, workSpaceSizeByte_backwardFilter)));

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);

		// �o�C�A�X�̃f�[�^�\����ݒ�
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->biasTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimBias[0],
			&dimBiasStride[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpConvolution_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// �w�K�W��
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"LearnCoeff"));
			if(pItem)
				this->learnData.LearnCoeff = pItem->GetValue();
			else
				this->learnData.LearnCoeff = 1.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpConvolution_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode UpConvolution_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���̓o�b�t�@��ۑ�
		this->m_lppInputBuffer_d = i_lpInputBuffer;

		// ���͐M�����g��
		{
		}

		// ��ݍ��ݏ���
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;
			err_cudnn = cudnnConvolutionForward(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				i_lpInputBuffer,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),
				this->convDesc,
				this->useForwardAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �o�C�A�X��ǉ�
		{
			F32 alpha = 1.0f;
			F32 beta  = 1.0f;

			err_cudnn = cudnnAddTensor(
				this->cudnnHandle,
				&alpha,
				this->biasTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),
				&beta,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER UpConvolution_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode UpConvolution_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode UpConvolution_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		cudnnStatus_t err_cudnn;

		// �o�͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lppDOutputBuffer_d = i_lpDOutputBufferPrev;

		// ���͌덷���v�Z
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardData(
				this->cudnnHandle,
				&alpha,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),
				this->outputTensorDesc,
				this->m_lppDOutputBuffer_d,
				this->convDesc,
				this->useBackwardDataAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->inputTensorDesc,
				thrust::raw_pointer_cast(&this->lpDInputBuffer[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �t�B���^�[�ω��ʂ��v�Z
		{
			F32 alpha = this->learnData.LearnCoeff;
			F32 beta  = 1.0f;

			err_cudnn = cudnnConvolutionBackwardFilter(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				this->m_lppInputBuffer_d,
				this->outputTensorDesc,
				this->m_lppDOutputBuffer_d,
				this->convDesc,
				this->useBackwardFilterAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// �o�C�A�X�ω��ʂ��v�Z
		{
			F32 alpha = this->learnData.LearnCoeff;
			F32 beta  = 1.0f;

			err_cudnn = cudnnConvolutionBackwardBias(
				this->cudnnHandle,
				&alpha,
				this->outputTensorDesc,
				this->m_lppDOutputBuffer_d,
				&beta,
				this->biasTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER UpConvolution_GPU::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode UpConvolution_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*this->batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;