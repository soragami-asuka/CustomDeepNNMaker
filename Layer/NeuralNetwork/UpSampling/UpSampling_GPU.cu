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
	UpSampling_GPU::UpSampling_GPU(Gravisbell::GUID guid, UpSampling_LayerData_GPU& i_layerData)
		:	UpSampling_Base	(guid)
		,	layerData			(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount	(0)		/**< ���̓o�b�t�@�� */
		,	outputBufferCount	(0)		/**< �o�̓o�b�t�@�� */
		,	cudnnHandle			(NULL)
		,	inputTensorDesc		(NULL)
		,	outputTensorDesc	(NULL)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&upscaleTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
	}
	/** �f�X�g���N�^ */
	UpSampling_GPU::~UpSampling_GPU()
	{
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(upscaleTensorDesc);
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
	ErrorCode UpSampling_GPU::PreProcessLearn(unsigned int batchSize)
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
	ErrorCode UpSampling_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

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
		std::vector<S32> dimUpScaleStride;
		if(this->layerData.inputDataStruct.z > 1)
		{
			dataDim = 1 + 1 + 3;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.z;
			dimInput[3] = this->layerData.inputDataStruct.y;
			dimInput[4] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->batchSize;
			dimOutput[1] = this->layerData.outputDataStruct.ch;
			dimOutput[2] = this->layerData.outputDataStruct.z;
			dimOutput[3] = this->layerData.outputDataStruct.y;
			dimOutput[4] = this->layerData.outputDataStruct.x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[2] = dimOutput[3] * dimOutput[4];
			dimOutputStride[3] = dimOutput[4];
			dimOutputStride[4] = 1;

			dimUpScaleStride.resize(dataDim);
			dimUpScaleStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3] * dimOutput[4] * this->layerData.layerStructure.UpScale.x;
			dimUpScaleStride[1] = dimOutput[2] * dimOutput[3] * dimOutput[4] * this->layerData.layerStructure.UpScale.x;
			dimUpScaleStride[2] = dimOutput[3] * dimOutput[4] * this->layerData.layerStructure.UpScale.x;
			dimUpScaleStride[3] = dimOutput[4] * this->layerData.layerStructure.UpScale.x;
			dimUpScaleStride[4] = this->layerData.layerStructure.UpScale.x;

			upscaleStride.x = 1;
			upscaleStride.y = this->layerData.layerStructure.UpScale.x * this->layerData.inputDataStruct.x;
			upscaleStride.z = this->layerData.layerStructure.UpScale.x * this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.y * this->layerData.inputDataStruct.y;
		}
		else if(this->layerData.inputDataStruct.y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.y;
			dimInput[3] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->batchSize;
			dimOutput[1] = this->layerData.outputDataStruct.ch;
			dimOutput[2] = this->layerData.outputDataStruct.y;
			dimOutput[3] = this->layerData.outputDataStruct.x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3];
			dimOutputStride[2] = dimOutput[3];
			dimOutputStride[3] = 1;
			
			dimUpScaleStride.resize(dataDim);
			dimUpScaleStride[0] = this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.x * this->layerData.inputDataStruct.y * this->layerData.layerStructure.UpScale.y * this->layerData.inputDataStruct.ch; 
			dimUpScaleStride[1] = this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.x * this->layerData.inputDataStruct.y * this->layerData.layerStructure.UpScale.y;
			dimUpScaleStride[2] = this->layerData.inputDataStruct.x * this->layerData.layerStructure.UpScale.x * this->layerData.layerStructure.UpScale.y;
			dimUpScaleStride[3] = this->layerData.layerStructure.UpScale.x;

			upscaleStride.x = 1;
			upscaleStride.y = this->layerData.layerStructure.UpScale.x * this->layerData.inputDataStruct.x;
			upscaleStride.z = 0;
		}
		else if(this->layerData.inputDataStruct.x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->batchSize;
			dimOutput[1] = this->layerData.outputDataStruct.ch;
			dimOutput[2] = this->layerData.outputDataStruct.x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2];
			dimOutputStride[1] = dimOutput[2];
			dimOutputStride[2] = 1;

			dimUpScaleStride.resize(dataDim);
			dimUpScaleStride[0] = dimOutput[1] * dimOutput[2] * this->layerData.layerStructure.UpScale.x;
			dimUpScaleStride[1] = dimOutput[2] * this->layerData.layerStructure.UpScale.x;
			dimUpScaleStride[2] = this->layerData.layerStructure.UpScale.x;
			
			upscaleStride.x = 1;
			upscaleStride.y = 0;
			upscaleStride.z = 0;
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

		// CUDNN�̓���>�o�͗p�f�[�^�\����ݒ�
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->upscaleTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimUpScaleStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;


		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpSampling_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode UpSampling_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode UpSampling_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// ���̓o�b�t�@��ۑ�
		this->m_lppInputBuffer_d = i_lpInputBuffer;

		// �o�̓o�b�t�@���N���A
		cudaMemset(
			thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
			0,
			this->lpOutputBuffer.size()*sizeof(F32));

		// ���̓o�b�t�@���o�͂ɃR�s�[
		switch(this->layerData.layerStructure.PaddingType)
		{
		case UpSampling::LayerStructure::PaddingType_value:
			{
				F32 alpha = 1.0f;
				F32 beta  = 1.0f;

				for(S32 offsetZ=0; offsetZ<this->layerData.layerStructure.UpScale.z; offsetZ++)
				{
					for(S32 offsetY=0; offsetY<this->layerData.layerStructure.UpScale.y; offsetY++)
					{
						for(S32 offsetX=0; offsetX<this->layerData.layerStructure.UpScale.x; offsetX++)
						{
							S32 offset = offsetX * this->upscaleStride.x + offsetY * this->upscaleStride.y + offsetZ * this->upscaleStride.z;

							err_cudnn = cudnnTransformTensor(
								this->cudnnHandle,
								&alpha,
								this->inputTensorDesc,
								this->m_lppInputBuffer_d,
								&beta,
								this->upscaleTensorDesc,
								thrust::raw_pointer_cast(&this->lpOutputBuffer[offset]));

							if(err_cudnn != 0)
								return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
						}
					}
				}
			}
			break;
		case UpSampling::LayerStructure::PaddingType_zero:
			{
				F32 alpha = 1.0f;
				F32 beta  = 0.0f;

				err_cudnn = cudnnTransformTensor(
					this->cudnnHandle,
					&alpha,
					this->inputTensorDesc,
					this->m_lppInputBuffer_d,
					&beta,
					this->upscaleTensorDesc,
					thrust::raw_pointer_cast(&this->lpOutputBuffer[0]));

				if(err_cudnn != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}
			break;
		}

#ifdef _DEBUG
		std::vector<F32> lpDebugInputBuffer(this->batchSize * this->inputBufferCount);
		cudaMemcpy(&lpDebugInputBuffer[0], this->m_lppInputBuffer_d, sizeof(F32)*lpDebugInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<F32> lpDebugOutputBuffer(this->lpOutputBuffer.size());
		cudaMemcpy(&lpDebugOutputBuffer[0], thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), sizeof(F32)*lpDebugOutputBuffer.size(), cudaMemcpyDeviceToHost);
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER UpSampling_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode UpSampling_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode UpSampling_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		cudnnStatus_t err_cudnn;

		// �o�͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lppDOutputBuffer_d = i_lpDOutputBufferPrev;

		// ���͌덷�o�b�t�@�̃N���A
		cudaMemset(
			thrust::raw_pointer_cast(&this->lpDInputBuffer[0]),
			0,
			this->lpDInputBuffer.size()*sizeof(F32));

		// �o�͌덷�o�b�t�@����͌덷�ɃR�s�[
		switch(this->layerData.layerStructure.PaddingType)
		{
		case UpSampling::LayerStructure::PaddingType_value:
			{
				F32 alpha = 1.0f / (this->layerData.layerStructure.UpScale.x * this->layerData.layerStructure.UpScale.y * this->layerData.layerStructure.UpScale.z);
				F32 beta  = 1.0f;

				for(S32 offsetZ=0; offsetZ<this->layerData.layerStructure.UpScale.z; offsetZ++)
				{
					for(S32 offsetY=0; offsetY<this->layerData.layerStructure.UpScale.y; offsetY++)
					{
						for(S32 offsetX=0; offsetX<this->layerData.layerStructure.UpScale.x; offsetX++)
						{
							S32 offset = offsetX * this->upscaleStride.x + offsetY * this->upscaleStride.y + offsetZ * this->upscaleStride.z;

//#ifdef _DEBUG
//							std::vector<F32> lpPreDebugDInputBuffer(this->lpDInputBuffer.size());
//							cudaMemcpy(&lpPreDebugDInputBuffer[0], thrust::raw_pointer_cast(&this->lpDInputBuffer[0]), sizeof(F32)*lpPreDebugDInputBuffer.size(), cudaMemcpyDeviceToHost);
//#endif

							err_cudnn = cudnnTransformTensor(
								this->cudnnHandle,
								&alpha,
								this->upscaleTensorDesc,
								&this->m_lppDOutputBuffer_d[offset],
								&beta,
								this->inputTensorDesc,
								thrust::raw_pointer_cast(&this->lpDInputBuffer[0]));

//#ifdef _DEBUG
//							std::vector<F32> lpDebugDOutputBuffer(this->lpOutputBuffer.size());
//							cudaMemcpy(&lpDebugDOutputBuffer[0], this->m_lppDOutputBuffer_d, sizeof(F32)*lpDebugDOutputBuffer.size(), cudaMemcpyDeviceToHost);
//		
//							std::vector<F32> lpDebugDInputBuffer(this->lpDInputBuffer.size());
//							cudaMemcpy(&lpDebugDInputBuffer[0], thrust::raw_pointer_cast(&this->lpDInputBuffer[0]), sizeof(F32)*lpDebugDInputBuffer.size(), cudaMemcpyDeviceToHost);
//#endif

							if(err_cudnn != 0)
								return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
							
						}
					}
				}
			}
			break;
		case UpSampling::LayerStructure::PaddingType_zero:
			{
				F32 alpha = 1.0f;
				F32 beta  = 0.0f;

				err_cudnn = cudnnTransformTensor(
					this->cudnnHandle,
					&alpha,
					this->upscaleTensorDesc,
					this->m_lppDOutputBuffer_d,
					&beta,
					this->inputTensorDesc,
					thrust::raw_pointer_cast(&this->lpDInputBuffer[0]));

				if(err_cudnn != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}
			break;
		}

#ifdef _DEBUG
		std::vector<F32> lpDebugDOutputBuffer(this->lpOutputBuffer.size());
		cudaMemcpy(&lpDebugDOutputBuffer[0], this->m_lppDOutputBuffer_d, sizeof(F32)*lpDebugDOutputBuffer.size(), cudaMemcpyDeviceToHost);
		
		std::vector<F32> lpDebugDInputBuffer(this->lpDInputBuffer.size());
		cudaMemcpy(&lpDebugDInputBuffer[0], thrust::raw_pointer_cast(&this->lpDInputBuffer[0]), sizeof(F32)*lpDebugDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER UpSampling_GPU::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode UpSampling_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
