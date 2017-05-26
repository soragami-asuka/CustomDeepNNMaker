//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"MaxAveragePooling_DATA.hpp"
#include"MaxAveragePooling_FUNC.hpp"
#include"MaxAveragePooling_Base.h"

#include"MaxAveragePooling_GPU.cuh"
#include"MaxAveragePooling_LayerData_GPU.cuh"

#ifndef __CUDACC__  
    #define __CUDACC__
#endif

#include<device_functions.hpp>

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(32)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	namespace
	{
		// ����p
		__global__ void cuda_func_average_input(const F32* i_lpInputBuffer, F32* o_lpOutputBuffer, const U32 i_inputChSize, U32 i_outputChSize)
		{
			const U32 batchNo = blockIdx.z;
			const U32 chNo    = blockIdx.y;
			const U32 chCount = blockDim.y;

			const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			const U32 outputPos = batchNo * (i_outputChSize * chCount) + chNo * i_outputChSize + bufferPos;
			const U32 inputPos  = batchNo * (i_inputChSize  * chCount) + chNo * i_inputChSize  + bufferPos;

			__shared__ F32 lpTmpBuf[BLOCK_SIZE*2];
			if(inputPos >= i_inputChSize)
				lpTmpBuf[blockIdx.x]  = 0.0f;
			else
				lpTmpBuf[blockIdx.x + 0]  = i_lpInputBuffer[inputPos];
			__syncthreads();

			if(threadIdx.x < 16)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 16];
			__syncthreads();
			if(threadIdx.x < 8)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 8];
			__syncthreads();
			if(threadIdx.x < 4)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 4];
			__syncthreads();
			if(threadIdx.x < 2)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 2];
			__syncthreads();
			if(threadIdx.x < 1)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 1];
			__syncthreads();

			o_lpOutputBuffer[outputPos] = lpTmpBuf[0];
		}
		// �r���v�Z�p(�y��)
		__global__ void cuda_func_average(const F32* i_lpInputBuffer, F32* o_lpOutputBuffer, const U32 i_inputChSize, U32 i_outputChSize)
		{
			const U32 batchNo = blockIdx.z;
			const U32 chNo    = blockIdx.y;
			const U32 chCount = blockDim.y;

			const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			const U32 outputPos = batchNo * (i_outputChSize * chCount) + chNo * i_outputChSize + bufferPos;
			const U32 inputPos  = batchNo * (i_inputChSize  * chCount) + chNo * i_inputChSize  + bufferPos;

			__shared__ F32 lpTmpBuf[BLOCK_SIZE*2];
			lpTmpBuf[blockIdx.x + 0]  = i_lpInputBuffer[inputPos];
			__syncthreads();

			if(threadIdx.x < 16)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 16];
			__syncthreads();
			if(threadIdx.x < 8)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 8];
			__syncthreads();
			if(threadIdx.x < 4)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 4];
			__syncthreads();
			if(threadIdx.x < 2)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 2];
			__syncthreads();
			if(threadIdx.x < 1)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 1];
			__syncthreads();

			o_lpOutputBuffer[outputPos] = lpTmpBuf[0];
		}
	}


	/** �R���X�g���N�^ */
	MaxAveragePooling_GPU::MaxAveragePooling_GPU(Gravisbell::GUID guid, MaxAveragePooling_LayerData_GPU& i_layerData)
		:	MaxAveragePooling_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
		,	m_lppInputBuffer				(NULL)			/**< ���Z���̓��̓f�[�^ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
	{
	}
	/** �f�X�g���N�^ */
	MaxAveragePooling_GPU::~MaxAveragePooling_GPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 MaxAveragePooling_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode MaxAveragePooling_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	MaxAveragePooling_LayerData_Base& MaxAveragePooling_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const MaxAveragePooling_LayerData_Base& MaxAveragePooling_GPU::GetLayerData()const
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
	ErrorCode MaxAveragePooling_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// ���͍����o�b�t�@���쐬
		this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MaxAveragePooling_GPU::PreProcessCalculate(unsigned int batchSize)
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

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);

		// 1CH������̃T�C�Y���v�Z
		this->chSize = this->GetInputDataStruct().x * this->GetInputDataStruct().y * this->GetInputDataStruct().z;

		// �ꎞ�o�b�t�@�̊m��
		this->lpTmpBuffer0.resize((this->chSize + 31)/32*32 * this->GetInputDataStruct().ch * this->batchSize, 0.0f);
		this->lpTmpBuffer1.resize((this->chSize + 31)/32*32 * this->GetInputDataStruct().ch * this->batchSize, 0.0f);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MaxAveragePooling_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode MaxAveragePooling_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode MaxAveragePooling_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X���i�[
		this->m_lppInputBuffer = i_lpInputBuffer;

		// ���񏈗�
		U32 tmpInputBufferCount = this->chSize;
		U32 tmpOutputBufferCount = (tmpInputBufferCount + (BLOCK_SIZE-1))/BLOCK_SIZE;
		{
			dim3 grid(tmpOutputBufferCount, this->GetInputDataStruct().ch, this->batchSize);

			cuda_func_average_input<<<grid, BLOCK_SIZE>>>(i_lpInputBuffer, thrust::raw_pointer_cast(&this->lpTmpBuffer0[0]), tmpInputBufferCount, tmpOutputBufferCount);
		}

		while(tmpOutputBufferCount > 1)
		{
			tmpInputBufferCount = tmpOutputBufferCount;
			tmpOutputBufferCount = (tmpInputBufferCount + (BLOCK_SIZE-1))/BLOCK_SIZE;

			cuda_func_average<<<grid, BLOCK_SIZE>>>(i_lpInputBuffer, thrust::raw_pointer_cast(&this->lpTmpBuffer0[0]), tmpInputBufferCount, tmpOutputBufferCount);
		}

		// �eCH�̗v�f��ch�T�C�Y�ŏ��Z���Ė{�̂Ɋi�[




		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER MaxAveragePooling_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode MaxAveragePooling_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode MaxAveragePooling_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lppDOutputBufferPrev = i_lppDOutputBufferPrev;


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER MaxAveragePooling_GPU::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode MaxAveragePooling_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
