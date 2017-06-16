//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// GPU�����p
//======================================
#include"stdafx.h"

#include"FullyConnect_DATA.hpp"
#include"FullyConnect_FUNC.hpp"
#include"FullyConnect_Base.h"

#include"FullyConnect_GPU.cuh"
#include"FullyConnect_LayerData_GPU.cuh"

#include"Library/NeuralNetwork/Optimizer.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(16)

namespace
{
	/** �x�N�g���̗v�f���m�̊|���Z. */
	__global__ void cuda_func_multiplVector(const F32* i_lpInputBufferA, const F32* i_lpInputBufferB, F32* o_lpOutputBuffer, U32 i_bufferSize)
	{
		const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(bufferPos >= i_bufferSize)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
			return;

		o_lpOutputBuffer[bufferPos] = i_lpInputBufferA[bufferPos] * i_lpInputBufferB[bufferPos];
	}
	/** �x�N�g���̗v�f���m�̊|���Z. */
	__global__ void cuda_func_multiplVectorWithScaler(const F32* i_lpInputBufferA, const F32* i_lpInputBufferB, F32* o_lpOutputBuffer, U32 i_bufferSize, F32 alpha, F32 beta)
	{
		const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(bufferPos >= i_bufferSize)	// ���򂷂邪������warp�����Ȃ̂ŁA�������x�ɉe���͂Ȃ��͂�...
			return;

		o_lpOutputBuffer[bufferPos] = alpha * i_lpInputBufferA[bufferPos] * i_lpInputBufferB[bufferPos] + beta * o_lpOutputBuffer[bufferPos];
	}
}

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** �R���X�g���N�^ */
	FullyConnect_GPU::FullyConnect_GPU(Gravisbell::GUID guid, FullyConnect_LayerData_GPU& i_layerData)
		:	FullyConnect_Base	(guid)
		,	layerData						(i_layerData)	/**< ���C���[�f�[�^ */
		,	inputBufferCount				(0)		/**< ���̓o�b�t�@�� */
		,	neuronCount						(0)		/**< �j���[������ */
		,	outputBufferCount				(0)		/**< �o�̓o�b�t�@�� */
	{
		cublasCreate(&cublasHandle);
	}
	/** �f�X�g���N�^ */
	FullyConnect_GPU::~FullyConnect_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// ��{����
	//================================
	/** ���C���[��ʂ̎擾 */
	U32 FullyConnect_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** ������. �e�j���[�����̒l�������_���ɏ�����
		@return	���������ꍇ0 */
	ErrorCode FullyConnect_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// ���C���[�f�[�^�֘A
	//===========================
	/** ���C���[�f�[�^���擾���� */
	FullyConnect_LayerData_Base& FullyConnect_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_LayerData_Base& FullyConnect_GPU::GetLayerData()const
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
	ErrorCode FullyConnect_GPU::PreProcessLearn(U32 batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�C�A�X�X�V�p�̃x�N�g�����쐬
		lpBiasUpdateVector_d.resize(this->batchSize);
		{
			thrust::host_vector<F32> lpBuf(this->batchSize, 1.0f);
			this->lpBiasUpdateVector_d = lpBuf;
		}

		// �p�����[�^�ω���
		this->lpDBias.resize(this->neuronCount);
		this->lpDNeuron.resize(this->neuronCount * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z�O���������s����.(���Z�p)
		@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
		NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_GPU::PreProcessCalculate(U32 batchSize)
	{
		this->batchSize = batchSize;

		// ���̓o�b�t�@�����m�F
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// �j���[���������m�F
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@�����m�F
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// �j���[�����o�b�t�@�̃T�C�Y�m�F
		if(this->layerData.lppNeuron_d.size() != this->neuronCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// �o�̓o�b�t�@���쐬
		this->lpOutputBuffer_d.resize(this->batchSize * this->outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();
		this->pLearnData->WriteToStruct((BYTE*)&this->learnData);

		switch(this->learnData.Optimizer)
		{
		case FullyConnect::LearnDataStructure::Optimizer_SGD:
			UpdateOptimizer_SGD_GPU(&this->m_pOptimizer_neuron, this->neuronCount*this->inputBufferCount, this->learnData.LearnCoeff);
			UpdateOptimizer_SGD_GPU(&this->m_pOptimizer_bias,   this->neuronCount,                        this->learnData.LearnCoeff);
			break;
		case FullyConnect::LearnDataStructure::Optimizer_Momentum:
			UpdateOptimizer_Momentum_GPU(&this->m_pOptimizer_neuron, this->neuronCount*this->inputBufferCount, this->learnData.LearnCoeff, this->learnData.Momentum_alpha);
			UpdateOptimizer_Momentum_GPU(&this->m_pOptimizer_bias,   this->neuronCount,                        this->learnData.LearnCoeff, this->learnData.Momentum_alpha);
			break;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FullyConnect_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// ���̓o�b�t�@��ۊ�
		this->m_lppInputBuffer_d = i_lpInputBuffer;

		// �o�C�A�X���o�͐M���ɃR�s�[����
		{
			for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
			{
				cudaError_t err = cudaMemcpy(
					thrust::raw_pointer_cast(&lpOutputBuffer_d[batchNum * this->outputBufferCount]),
					thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),
					sizeof(F32) * this->neuronCount,
					cudaMemcpyDeviceToDevice);
				if(err != 0)
					return ERROR_CODE_CUDA_COPY_MEMORY;
			}
		}

		// �j���[����T�~���͐M��
		{
			// C = aAB + bC;

			F32 alpha = 1.0f;
			F32 beta  = 1.0f;	// �o�C�A�X��C�ɃR�s�[�ς݂Ȃ̂ł��̂܂ܗ��p���邽�߂�1.0���w��

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				this->neuronCount,	// �s��A�̍s��
				this->batchSize,	// �s��B�̗�
				this->inputBufferCount,	// �s��A�̗�,�s��B�̍s��
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// �s��A
				this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
				i_lpInputBuffer,											// �s��B
				this->inputBufferCount,										// �s��B�̓]�u�O�̍s��
				&beta,
				thrust::raw_pointer_cast(&lpOutputBuffer_d[0]),
				this->outputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �o�̓f�[�^�o�b�t�@���擾����.
		�z��̗v�f����GetOutputBufferCount�̖߂�l.
		@return �o�̓f�[�^�z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer_d[0]);
	}
	/** �o�̓f�[�^�o�b�t�@���擾����.
		@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
		@return ���������ꍇ0 */
	ErrorCode FullyConnect_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32) * outputBufferCount * this->batchSize, cudaMemcpyDeviceToHost);

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
	ErrorCode FullyConnect_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// �o�͌덷�o�b�t�@�̃A�h���X���i�[
		this->m_lppDOutputBuffer_d = i_lppDOutputBuffer;
		// ���͌덷�o�b�t�@�̃A�h���X��z��Ɋi�[
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;

		// ���͌덷�������v�Z
		if(this->m_lpDInputBuffer_d)
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->inputBufferCount,	// �s��A�̍s��
				this->batchSize,		// �s��B�̗�
				this->neuronCount,		// �s��A�̗�,�s��B�̍s��
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// �s��A
				this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
				this->m_lppDOutputBuffer_d,									// �s��B
				this->neuronCount,											// �s��B�̓]�u�O�̍s��
				&beta,
				this->m_lpDInputBuffer_d,
				this->inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FullyConnect_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�v�Z
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		
		// �o�C�A�X�ω��ʌv�Z
		{
			F32 alpha = 1.0f;
			F32 beta  = 0;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->neuronCount,		// �s��A�̍s��
				1,						// �s��B�̗�
				this->batchSize,		// �s��A�̗�,�s��B�̍s��
				&alpha,
				this->m_lppDOutputBuffer_d,	// �s��A
				this->neuronCount,											// �s��A�̓]�u�O�̍s��
				thrust::raw_pointer_cast(&this->lpBiasUpdateVector_d[0]),	// �s��B
				this->batchSize,											// �s��B�̓]�u�O�̍s��
				&beta,
				thrust::raw_pointer_cast(&this->lpDBias[0]),
				this->neuronCount);
		}

		// �j���[�����ω��ʌv�Z
		{
			// �j���[�����̌덷���v�Z���ĉ��Z����
			{
				F32 alpha = 1.0f;
				F32 beta  = 0;

				cublasSgemm(
					this->cublasHandle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					this->inputBufferCount,	// �s��A�̍s��
					this->neuronCount,		// �s��B�̗�
					this->batchSize,		// �s��A�̗�,�s��B�̍s��
					&alpha,
					this->m_lppInputBuffer_d,		// �s��A
					this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
					this->m_lppDOutputBuffer_d,	// �s��B
					this->neuronCount,										// �s��B�̓]�u�O�̍s��
					&beta,
					thrust::raw_pointer_cast(&this->lpDNeuron[0]),
					this->inputBufferCount);
			}
		}


		// �덷�𔽉f
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),   thrust::raw_pointer_cast(&this->lpDBias[0]));
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]), thrust::raw_pointer_cast(&this->lpDNeuron[0]));


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** �w�K�������擾����.
		�z��̗v�f����[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]
		@return	�덷�����z��̐擪�|�C���^ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** �w�K�������擾����.
		@param lpDInputBuffer	�w�K�������i�[����z��.[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̔z�񂪕K�v */
	ErrorCode FullyConnect_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32) * inputBufferCount * batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
