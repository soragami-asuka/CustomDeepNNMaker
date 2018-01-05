//======================================
// ��ݍ��݃j���[�����l�b�g���[�N�̌������C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"UpSampling_DATA.hpp"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_Base.h"

#include"UpSampling_GPU.cuh"
#include"UpSampling_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	UpSampling_GPU::UpSampling_GPU(Gravisbell::GUID guid, UpSampling_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	UpSampling_Base		(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData			(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount	(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount	(0)		/**< �o�̓o�b�t�@�� */
		,	cudnnHandle			(NULL)
		,	inputTensorDesc		(NULL)
		,	outputTensorDesc	(NULL)
		,	filterDesc			(NULL)
		,	convDesc			(NULL)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateFilterDescriptor(&filterDesc);
		cudnnCreateConvolutionDescriptor(&convDesc);
	}
	/** �f�X�g���N�^ */
	UpSampling_GPU::~UpSampling_GPU()
	{
		if(convDesc)			cudnnDestroyConvolutionDescriptor(convDesc);
		if(filterDesc)			cudnnDestroyFilterDescriptor(filterDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 UpSampling_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode UpSampling_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	UpSampling_LayerData_Base& UpSampling_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const UpSampling_LayerData_Base& UpSampling_GPU::GetLayerData()const
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
	ErrorCode UpSampling_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpSampling_GPU::PreProcessCalculate()
	{
		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		cudnnStatus_t err_cudnn;

		// �������𒲂ׂ�
		S32 dataDim = 1 + 1 + 0;	// �o�b�` + �`�����l�� + ����0
		std::vector<S32> dimInput;			// ���̓f�[�^�\��
		std::vector<S32> dimInputStride;	// ���̓f�[�^�̊e�������Ƃ̃f�[�^��
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		S32 filterDim = 0;			// �t�B���^������	���̓`�����l�� + �o�̓`�����l�� + ����
		std::vector<S32> dimFilter;
		S32 convDim = 0;			// ��ݍ��ݎ�����	����
		std::vector<S32> dimStride;
		std::vector<S32> dimDilation;
		std::vector<S32> dimPadding;
		if(this->GetInputDataStruct().z > 1)
		{
			dataDim = 1 + 1 + 3;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().z;
			dimInput[3] = this->GetInputDataStruct().y;
			dimInput[4] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().z;
			dimOutput[3] = this->GetOutputDataStruct().y;
			dimOutput[4] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[2] = dimOutput[3] * dimOutput[4];
			dimOutputStride[3] = dimOutput[4];
			dimOutputStride[4] = 1;

			filterDim = 1 + 1 + 2;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.UpScale.y;
			dimFilter[3] = this->layerData.layerStructure.UpScale.x;

			convDim = 2;	// ����3

			dimPadding.resize(convDim);
			dimPadding[0] = 0;
			dimPadding[1] = 0;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;
			dimDilation[1] = 1;

			dimStride.resize(convDim);
			dimStride[0] = 1;
			dimStride[1] = 1;

		}
		else if(this->GetInputDataStruct().y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize() * this->GetInputDataStruct().ch;
			dimInput[1] = 1;
			dimInput[2] = this->GetInputDataStruct().y;
			dimInput[3] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize() * this->GetOutputDataStruct().ch;
			dimOutput[1] = 1;
			dimOutput[2] = this->GetOutputDataStruct().y;
			dimOutput[3] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3];
			dimOutputStride[2] = dimOutput[3];
			dimOutputStride[3] = 1;

			filterDim = 1 + 1 + 2;	// ���̓`�����l�� + �o�̓`�����l�� + ����3

			dimFilter.resize(filterDim);
			dimFilter[0] = 1;
			dimFilter[1] = 1;
			dimFilter[2] = this->layerData.layerStructure.UpScale.y;
			dimFilter[3] = this->layerData.layerStructure.UpScale.x;

			convDim = 2;	// ����2

			dimPadding.resize(convDim);
			dimPadding[0] = 0;
			dimPadding[1] = 0;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;
			dimDilation[1] = 1;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.UpScale.y;
			dimStride[1] = this->layerData.layerStructure.UpScale.x;
		}
		else if(this->GetInputDataStruct().x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2];
			dimOutputStride[1] = dimOutput[2];
			dimOutputStride[2] = 1;

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

		// CUDNN�̏o�̓f�[�^�\����ݒ�
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
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

		// �ő��̃A���S���Y������������(�O���`�d)
		err_cudnn = cudnnGetConvolutionForwardAlgorithm(
			this->cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
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
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->useForwardAlgorithm,
			&workSpaceSizeByte_forward);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �ő��̃A���S���Y������������(����`�d-�f�[�^)
		err_cudnn = cudnnGetConvolutionBackwardDataAlgorithm(
			this->cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
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
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->useBackwardDataAlgorithm,
			&workSpaceSizeByte_backwardData);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// �����p�o�b�t�@�̊m��
		this->workSpace.resize(max(workSpaceSizeByte_forward, workSpaceSizeByte_backwardData));


		// �t�B���^�o�b�t�@���쐬���ď�����
		filter.resize(
			this->layerData.layerStructure.UpScale.x * this->layerData.layerStructure.UpScale.y * this->layerData.layerStructure.UpScale.z,
			0.0f);
		for(U32 z=0; z<this->layerData.layerStructure.UpScale.z; z++)
		{
			U32 zOffset = z * this->layerData.layerStructure.UpScale.y * this->layerData.layerStructure.UpScale.x;

			for(U32 y=0; y<this->layerData.layerStructure.UpScale.y; y++)
			{
				U32 yOffset = y * this->layerData.layerStructure.UpScale.x;

				for(U32 x=0; x<this->layerData.layerStructure.UpScale.x; x++)
				{
					U32 offset = zOffset + yOffset + x;

					switch(this->layerData.layerStructure.PaddingType)
					{
					case UpSampling::LayerStructure::PaddingType_value:
						{
							filter[offset] = 1.0f;
						}
						break;
					case UpSampling::LayerStructure::PaddingType_zero:
						{
							if(z==0 && y==0 && x==0)
								filter[offset] = 1.0f;
							else
								filter[offset] = 0.0f;
						}
						break;
					}
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpSampling_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode UpSampling_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// �o�̓o�b�t�@���N���A
		cudaMemset(
			o_lppOutputBuffer,
			0,
			this->outputBufferCount * this->GetBatchSize() * sizeof(F32));

		// ���̓o�b�t�@���o�͂ɃR�s�[
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardData(
				this->cudnnHandle,
				&alpha,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->filter[0]),
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->convDesc,
				this->useBackwardDataAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->outputTensorDesc,
				o_lppOutputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

#ifdef _DEBUG
		std::vector<F32> lpDebugInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpDebugInputBuffer[0], i_lppInputBuffer, sizeof(F32)*lpDebugInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<F32> lpDebugOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpDebugOutputBuffer[0], o_lppOutputBuffer, sizeof(F32)*lpDebugOutputBuffer.size(), cudaMemcpyDeviceToHost);
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// �w�K����
	//================================
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode UpSampling_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			// ���͌덷�o�b�t�@�̃N���A
			cudaMemset(
				o_lppDInputBuffer,
				0,
				sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

			{
				F32 alpha = 1.0f;
				F32 beta  = 0.0f;
				err_cudnn = cudnnConvolutionForward(
					this->cudnnHandle,
					&alpha,
					this->outputTensorDesc,
					i_lppDOutputBuffer,
					this->filterDesc,
					thrust::raw_pointer_cast(&this->filter[0]),
					this->convDesc,
					this->useForwardAlgorithm,
					thrust::raw_pointer_cast(&this->workSpace[0]),
					this->workSpace.size(),
					&beta,
					this->inputTensorDesc,
					o_lppDInputBuffer);
				if(err_cudnn != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode UpSampling_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
