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
	FullyConnect_GPU::FullyConnect_GPU(Gravisbell::GUID guid, FullyConnect_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FullyConnect_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
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
	ErrorCode FullyConnect_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// �o�C�A�X�X�V�p�̃x�N�g�����쐬
		lpBiasUpdateVector_d.resize(this->GetBatchSize());
		{
			thrust::host_vector<F32> lpBuf(this->GetBatchSize(), 1.0f);
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
	ErrorCode FullyConnect_GPU::PreProcessCalculate()
	{
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

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
		���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
	ErrorCode FullyConnect_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ���Z���������s����.
		@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
		@return ���������ꍇ0���Ԃ� */
	ErrorCode FullyConnect_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			// ���Ƃ肠����CPU���ŏ���.
			// ��{�I��1�񂵂��ʂ�Ȃ����珈�����ׂɉe���͗^���Ȃ��E�E�E�͂�
			// ���蔲��


			U32 PROCTIME_MAX = 5;			// ���s�ő�l
			F32	VARIANCE_TOLERANCE = 0.1f;	// ���U����(���e�͈�)

			std::vector<F32> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);

			U32 procTime = 0;
			do
			{
				// ���Z�����s
				ErrorCode err = this->CalculateBase(i_lppInputBuffer, o_lppOutputBuffer);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				// �o�b�t�@���R�s�[
				cudaMemcpy(&lpTmpOutputBuffer[0], &o_lppOutputBuffer[0], sizeof(F32)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

				// �o�͂̕��U�����߂�
				F32 variance = 0.0f;
				F32 average  = 0.0f;
				{
					// ���ς����߂�
					for(U32 outputNum=0; outputNum<lpTmpOutputBuffer.size(); outputNum++)
					{
						average += lpTmpOutputBuffer[outputNum];
					}
					average /= lpTmpOutputBuffer.size();

					// ���U�����߂�
					for(U32 outputNum=0; outputNum<lpTmpOutputBuffer.size(); outputNum++)
					{
						variance += (lpTmpOutputBuffer[outputNum] - average) * (lpTmpOutputBuffer[outputNum] - average);
					}
					variance /= lpTmpOutputBuffer.size();
				}

				if( abs(variance - 1.0f) < VARIANCE_TOLERANCE)
					break;

				// �W���΍��ŏd�݂������čX�V����
				F32 deviation = sqrtf(variance);
				{
					thrust::host_vector<F32> lpTmpNeuron = this->layerData.lppNeuron_d;
					thrust::host_vector<F32> lpTmpBias   = this->layerData.lpBias_d;

					for(U32 neuronNum=0; neuronNum<lpTmpNeuron.size(); neuronNum++)
					{
						lpTmpNeuron[neuronNum] /= deviation;
					}
					for(U32 neuronNum=0; neuronNum<lpTmpBias.size(); neuronNum++)
					{
						lpTmpBias[neuronNum] /= deviation;
					}

					this->layerData.lppNeuron_d = lpTmpNeuron;
					this->layerData.lpBias_d    = lpTmpBias;
				}

				procTime++;
			}while(procTime < 5);
		}
		else
		{
			ErrorCode err = this->CalculateBase(i_lppInputBuffer, o_lppOutputBuffer);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FullyConnect_GPU::CalculateBase(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// �o�C�A�X���o�͐M���ɃR�s�[����
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				cudaError_t err = cudaMemcpy(
					&o_lppOutputBuffer[batchNum * this->outputBufferCount],
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
				this->GetBatchSize(),	// �s��B�̗�
				this->inputBufferCount,	// �s��A�̗�,�s��B�̍s��
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// �s��A
				this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
				i_lppInputBuffer,											// �s��B
				this->inputBufferCount,										// �s��B�̓]�u�O�̍s��
				&beta,
				&o_lppOutputBuffer[0],
				this->outputBufferCount);
		}

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
	ErrorCode FullyConnect_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// ���͌덷�������v�Z
		if(o_lppDInputBuffer)
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->inputBufferCount,	// �s��A�̍s��
				this->GetBatchSize(),		// �s��B�̗�
				this->neuronCount,		// �s��A�̗�,�s��B�̍s��
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// �s��A
				this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
				i_lppDOutputBuffer,											// �s��B
				this->neuronCount,											// �s��B�̓]�u�O�̍s��
				&beta,
				o_lppDInputBuffer,
				this->inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** �w�K���������s����.
		���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
		@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
		���O�̌v�Z���ʂ��g�p���� */
	ErrorCode FullyConnect_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		std::vector<F32> lpDOutputBuffer_h(this->outputBufferCount * this->GetBatchSize());
		cudaMemcpy(&lpDOutputBuffer_h[0], i_lppDOutputBuffer, sizeof(F32)*lpDOutputBuffer_h.size(), cudaMemcpyDeviceToHost);

		std::vector<F32> lpInputBuffer_h(this->inputBufferCount * this->GetBatchSize());
		cudaMemcpy(&lpInputBuffer_h[0], i_lppInputBuffer, sizeof(F32)*lpInputBuffer_h.size(), cudaMemcpyDeviceToHost);


		// ���͌덷�v�Z
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
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
				this->GetBatchSize(),		// �s��A�̗�,�s��B�̍s��
				&alpha,
				i_lppDOutputBuffer,		// �s��A
				this->neuronCount,											// �s��A�̓]�u�O�̍s��
				thrust::raw_pointer_cast(&this->lpBiasUpdateVector_d[0]),	// �s��B
				this->GetBatchSize(),										// �s��B�̓]�u�O�̍s��
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
					this->GetBatchSize(),		// �s��A�̗�,�s��B�̍s��
					&alpha,
					i_lppInputBuffer,		// �s��A
					this->inputBufferCount,										// �s��A�̓]�u�O�̍s��
					i_lppDOutputBuffer,	// �s��B
					this->neuronCount,										// �s��B�̓]�u�O�̍s��
					&beta,
					thrust::raw_pointer_cast(&this->lpDNeuron[0]),
					this->inputBufferCount);
			}
		}


		// �덷�𔽉f
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),   thrust::raw_pointer_cast(&this->lpDBias[0]));
		if(this->layerData.m_pOptimizer_neuron)
			this->layerData.m_pOptimizer_neuron->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]), thrust::raw_pointer_cast(&this->lpDNeuron[0]));

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
