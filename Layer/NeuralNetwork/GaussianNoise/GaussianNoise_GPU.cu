//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"GaussianNoise_DATA.hpp"
#include"GaussianNoise_FUNC.hpp"
#include"GaussianNoise_Base.h"

#include"GaussianNoise_GPU.cuh"
#include"GaussianNoise_LayerData_GPU.cuh"

#include<curand.h>
#include<curand_kernel.h>

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define THREAD_EXEC_SIZE	(32)
#define BLOCK_SIZE			(32)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	__global__ void RandomGenerator(U32 i_seed, F32 average, F32 variance, F32 o_lpOutput[], int bufferSize)
	{
		S32 id = blockIdx.x * blockDim.x + threadIdx.x;
		curandState s;

		curand_init(i_seed, id, 0, &s);

		for(S32 i=0; i<THREAD_EXEC_SIZE; i++)
		{
			S32 pos = id*THREAD_EXEC_SIZE + i;

			if(pos >= bufferSize)
				break;

			// Box-Muller
			F32 alpha = curand_uniform(&s);
			F32 beta  = curand_uniform(&s);;
			F32 randomValue = sqrtf(-2.0f * log(alpha)) * sinf(2.0f * 3.1415f * beta);

			o_lpOutput[pos] += randomValue * variance + average;
		}
	}

	/** �R���X�g���N�^ */
	GaussianNoise_GPU::GaussianNoise_GPU(Gravisbell::GUID guid, GaussianNoise_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	GaussianNoise_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)				/**< ���̓o�b�t�@�� */
		,	outputBufferCount				(0)				/**< �o�̓o�b�t�@�� */
		,	m_lppInputBuffer				(NULL)			/**< ���Z���̓��̓f�[�^ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< ���͌덷�v�Z���̏o�͌덷�f�[�^ */
	{
	}
	/** �f�X�g���N�^ */
	GaussianNoise_GPU::~GaussianNoise_GPU()
	{
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 GaussianNoise_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode GaussianNoise_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	GaussianNoise_LayerData_Base& GaussianNoise_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const GaussianNoise_LayerData_Base& GaussianNoise_GPU::GetLayerData()const
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
	ErrorCode GaussianNoise_GPU::PreProcessLearn()
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
	ErrorCode GaussianNoise_GPU::PreProcessCalculate()
	{
		cudnnStatus_t err_cudnn;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}
	


	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode GaussianNoise_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode GaussianNoise_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@�̃A�h���X���i�[
		this->m_lppInputBuffer = i_lpInputBuffer;

		// ���̓o�b�t�@���o�̓o�b�t�@�ɃR�s�[
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), i_lpInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);

		F32 average  = this->layerData.layerStructure.Average  + this->GetRuntimeParameterByStructure().GaussianNoise_Bias;
		F32 variance = this->layerData.layerStructure.Variance * this->GetRuntimeParameterByStructure().GaussianNoise_Power;

		// �m�C�Y�����Z
		dim3 grid((this->outputBufferCount*this->GetBatchSize() + (BLOCK_SIZE*THREAD_EXEC_SIZE-1)) / (BLOCK_SIZE*THREAD_EXEC_SIZE), 1 , 1);
		dim3 block(BLOCK_SIZE);
		RandomGenerator<<<grid, block>>>(0, average, variance, thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), this->outputBufferCount*this->GetBatchSize());

#ifdef _DEBUG
		std::vector<F32> lpTmpInputBuffer(this->inputBufferCount*this->GetBatchSize());
		cudaMemcpy(&lpTmpInputBuffer[0], i_lpInputBuffer, sizeof(F32)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<F32> lpTmpOutputBuffer(this->outputBufferCount*this->GetBatchSize());
		cudaMemcpy(&lpTmpOutputBuffer[0], thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), sizeof(F32)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GaussianNoise_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode GaussianNoise_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode GaussianNoise_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// ���͌덷�v�Z
		if(o_lppDInputBuffer)
		{
			cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode GaussianNoise_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER GaussianNoise_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode GaussianNoise_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
