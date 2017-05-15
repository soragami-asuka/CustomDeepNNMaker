//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// CPU�����p
//======================================
#include"stdafx.h"

#include"Dropout_DATA.hpp"
#include"Dropout_FUNC.hpp"
#include"Dropout_Base.h"

#include"Dropout_GPU.cuh"
#include"Dropout_LayerData_GPU.cuh"

#include<time.h>

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	Dropout_GPU::Dropout_GPU(Gravisbell::GUID guid, Dropout_LayerData_GPU& i_layerData)
		:	Dropout_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
		,	onLearning						(false)			/**< �w�K�������t���O */
		,	cudnnHandle		(NULL)
		,	dropoutDesc		(NULL)
		,	inputTensorDesc	(NULL)
		,	outputTensorDesc	(NULL)
		,	m_pState		(NULL)
		,	m_pReserve		(NULL)
		,	reserveSize		(0)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateDropoutDescriptor(&dropoutDesc);
	}
	/** �f�X�g���N�^ */
	Dropout_GPU::~Dropout_GPU()
	{
		if(this->m_pState)		cudaFree(this->m_pState);
		if(this->m_pReserve)	cudaFree(this->m_pReserve);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(dropoutDesc)			cudnnDestroyDropoutDescriptor(dropoutDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 Dropout_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode Dropout_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	Dropout_LayerData_Base& Dropout_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Dropout_LayerData_Base& Dropout_GPU::GetLayerData()const
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
	ErrorCode Dropout_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �w�K�������t���O��ݒ肷��
		this->onLearning = true;

		// ���͍����o�b�t�@���쐬
		this->lpDInputBuffer_d.resize(this->batchSize * this->inputBufferCount);
		
		// �o�̓o�b�t�@���쐬
		{
			int n = this->batchSize;
			int c = this->GetOutputDataStruct().ch;
			int h = this->GetOutputDataStruct().z * this->GetOutputDataStruct().y;
			int w = this->GetOutputDataStruct().x;

			const int nDims = 4;
			int dimA[nDims] = {n, c, h, w};
			int strideA[nDims] = {c*h*w, h*w, w, 1};

			cudnnStatus_t err = cudnnSetTensorNdDescriptor(this->outputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

			err = cudnnSetTensorNdDescriptor(this->inputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

			this->lpOutputBuffer_d.resize(this->batchSize * this->inputBufferCount);
		}

		// �h���b�v�A�E�g�ݒ���쐬
		{
			// �����W�F�l���[�^�p���������m��
			size_t stateSize = 0;
			{
				if(this->m_pState != NULL)
				{
					cudaFree(this->m_pState);
					this->m_pState = NULL;
				}

				cudnnDropoutGetStatesSize(this->cudnnHandle, &stateSize);

				cudaError err = cudaMalloc((void**)&this->m_pState, stateSize);
				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;
			}
			
			// �h���b�v�A�E�g�o�b�t�@�p���������m��
			{
				if(this->m_pReserve != NULL)
				{
					cudaFree(this->m_pReserve);
					this->m_pReserve = NULL;
				}

				cudnnStatus_t cudnnErr = cudnnDropoutGetReserveSpaceSize(this->inputTensorDesc, &this->reserveSize);
				if(cudnnErr != 0)
					return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

				cudaError cudaErr = cudaMalloc((void**)&this->m_pReserve, this->reserveSize);
				if(cudaErr != 0)
					return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;
			}

			// �h���b�v�A�E�g�ݒ���쐬
			{
				cudnnStatus_t err = cudnnSetDropoutDescriptor(
					this->dropoutDesc,
					this->cudnnHandle,
					this->layerData.layerStructure.Rate,
					this->m_pState,
					stateSize,
					(U64)time(NULL));

				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Dropout_GPU::PreProcessCalculate(unsigned int batchSize)
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

		// �w�K�������t���O���~�낷
		this->onLearning = false;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Dropout_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode Dropout_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode Dropout_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpInputBuffer_d = i_lpInputBuffer;

		if(this->onLearning)
		{
			cudnnStatus_t err = cudnnDropoutForward(
				this->cudnnHandle,
				this->dropoutDesc,
				this->inputTensorDesc,
				this->m_lpInputBuffer_d,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer_d[0]),
				this->m_pReserve,
				this->reserveSize);

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Dropout_GPU::GetOutputBuffer()const
	{
		if(this->onLearning)
		{
			return thrust::raw_pointer_cast(&this->lpOutputBuffer_d[0]);
		}
		else
		{
			return this->m_lpInputBuffer_d;
		}
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode Dropout_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode Dropout_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpDOutputBufferPrev_d = i_lpDOutputBufferPrev;

		if(this->onLearning)
		{
			cudnnStatus_t err = cudnnDropoutBackward(
				this->cudnnHandle,
				this->dropoutDesc,
				this->outputTensorDesc,
				i_lpDOutputBufferPrev,
				this->inputTensorDesc,
				thrust::raw_pointer_cast(&this->lpDInputBuffer_d[0]),
				this->m_pReserve,
				this->reserveSize);

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER Dropout_GPU::GetDInputBuffer()const
	{
		if(this->onLearning)
		{
			return thrust::raw_pointer_cast(&this->lpDInputBuffer_d[0]);
		}
		else
		{
			return this->m_lpDOutputBufferPrev_d;
		}
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode Dropout_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
