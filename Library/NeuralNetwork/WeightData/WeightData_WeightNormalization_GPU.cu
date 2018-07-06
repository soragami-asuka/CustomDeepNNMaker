//=====================================
// �d�݃f�[�^�N���X.CPU����
// �f�t�H���g.
//=====================================
#include"stdafx.h"

#include<vector>

#pragma warning(push)
#pragma warning(disable : 4267)
#pragma warning(disable : 4819)
#include <cuda.h> // need CUDA_VERSION
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <device_launch_parameters.h>
#pragma warning(pop)

#include"WeightData_WeightNormalization.h"

#include<Layer/NeuralNetwork/IOptimizer.h>
#include<Library/NeuralNetwork/Optimizer.h>
#include<Library/NeuralNetwork/Initializer.h>

namespace
{
#define CALCULATE_DSCALE_BLOCK_SIZE	32

	using namespace Gravisbell;

	__global__ void device_ResetParameter(F32* o_lpScale, F32* o_lpVector, F32* o_lpVectorScale, const F32* i_lpWeight, U32 i_inputCount, U32 i_loopCount)
	{
		__shared__ F32 lpTmpScale[CALCULATE_DSCALE_BLOCK_SIZE *2];

		U32 neuronNum = blockIdx.x;
		U32 tid = threadIdx.x;

		// �X�P�[�����v�Z
		lpTmpScale[tid] = 0.0f;
		lpTmpScale[tid + CALCULATE_DSCALE_BLOCK_SIZE] = 0.0f;
		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 inputNum = CALCULATE_DSCALE_BLOCK_SIZE * loopNum + tid;
			U32 offset = neuronNum * i_inputCount + inputNum;

			lpTmpScale[tid] += (inputNum < i_inputCount) ? i_lpWeight[offset] * i_lpWeight[offset] : 0.0f;
		}
		__syncthreads();

		// ���v
		lpTmpScale[tid] += lpTmpScale[tid + 16];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 8];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 4];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 2];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 1];
		__syncthreads();

		if(tid == 0)
			o_lpScale[neuronNum] = sqrtf(lpTmpScale[tid]);
		__syncthreads();

		// �x�N�^�[�T�C�Y��1�ɂ���
		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 inputNum = CALCULATE_DSCALE_BLOCK_SIZE * loopNum + tid;
			U32 offset = neuronNum * i_inputCount + inputNum;

			o_lpVector[offset] = i_lpWeight[offset] / o_lpScale[neuronNum];
		}

		// �x�N�^�[�X�P�[����1�ɂ���
		o_lpVectorScale[neuronNum] = 1.0f;
	}

	__global__ void device_UpdateWeight(F32* o_lpWeight, const F32* i_lpScale, const F32* i_lpVector, const F32* i_lpVectorScale)
	{
		U32 neuronNum = blockIdx.x;
		U32 inputNum  = threadIdx.x;

		U32 inputCount = blockDim.x;

		o_lpWeight[neuronNum * inputCount + inputNum] = i_lpScale[neuronNum] * i_lpVector[neuronNum*inputCount + inputNum] / i_lpVectorScale[neuronNum];
	}
	__global__ void device_UpdateWeight_v2(F32* o_lpWeight, const F32* i_lpScale, const F32* i_lpVector, const F32* i_lpVectorScale, U32 i_inputCount, U32 i_loopCount)
	{
		U32 neuronNum = blockIdx.x;
		U32 tid = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 inputNum = CALCULATE_DSCALE_BLOCK_SIZE * loopNum + tid;
			U32 offset = neuronNum * i_inputCount + inputNum;
			
			o_lpWeight[neuronNum * i_inputCount + inputNum] = i_lpScale[neuronNum] * i_lpVector[neuronNum*i_inputCount + inputNum] / i_lpVectorScale[neuronNum];
		}
	}

	__global__ void device_CalculateDScale(F32* o_lpDScale, const F32* i_lpDWeight, const F32* i_lpVector, const F32* i_lpVectorScale, U32 i_inputCount, U32 i_loopCount)
	{
		__shared__ F32 lpTmpScale[CALCULATE_DSCALE_BLOCK_SIZE *2];

		U32 neuronNum = blockIdx.x;
		U32 tid = threadIdx.x;

		// DWeight��Vector�̏�Z���v�Z
		lpTmpScale[tid] = 0.0f;
		lpTmpScale[tid + CALCULATE_DSCALE_BLOCK_SIZE] = 0.0f;
		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 inputNum = CALCULATE_DSCALE_BLOCK_SIZE * loopNum + tid;
			U32 offset = neuronNum * i_inputCount + inputNum;

			lpTmpScale[tid] += (inputNum < i_inputCount) ? i_lpDWeight[offset] * i_lpVector[offset] : 0.0f;
		}
		__syncthreads();

		// ���v
		lpTmpScale[tid] += lpTmpScale[tid + 16];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 8];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 4];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 2];
		__syncthreads();
		lpTmpScale[tid] += lpTmpScale[tid + 1];
		__syncthreads();

		if(tid == 0)
			o_lpDScale[neuronNum] = lpTmpScale[tid] / i_lpVectorScale[neuronNum];
	}

	__global__ void device_CalculateDVector(F32* o_lpDVector, const F32* i_lpScale, const F32* i_lpVector, const F32* i_lpVectorScale, const F32* i_lpDWeight, const F32* i_lpDScale)
	{
		U32 neuronNum = blockIdx.x;
		U32 inputNum  = threadIdx.x;

		U32 inputCount = blockDim.x;

		U32 offset = neuronNum * inputCount + inputNum;

		o_lpDVector[offset] = (i_lpScale[neuronNum] / i_lpVectorScale[neuronNum]) * (i_lpDWeight[offset] - i_lpDScale[neuronNum]*i_lpVector[offset]/i_lpVectorScale[neuronNum]);
	}
	
	__global__ void device_CalculateScale(F32* o_lpScale, const F32* i_lpVector, U32 i_inputCount, U32 i_loopCount)
	{
		__shared__ F32 lpTmpScale0[CALCULATE_DSCALE_BLOCK_SIZE*2];

		U32 neuronNum = blockIdx.x;
		U32 tid = threadIdx.x;

		// DWeight��Vector�̏�Z���v�Z
		lpTmpScale0[tid] = 0.0f;
		lpTmpScale0[tid + CALCULATE_DSCALE_BLOCK_SIZE] = 0.0f;
		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 inputNum = CALCULATE_DSCALE_BLOCK_SIZE * loopNum + tid;
			U32 offset = neuronNum * i_inputCount + inputNum;

			lpTmpScale0[tid] += (inputNum < i_inputCount)  ? i_lpVector[offset] * i_lpVector[offset] : 0.0f;
		}
		__syncthreads();

		// ���v
		if(tid < 16)
			lpTmpScale0[tid] += lpTmpScale0[tid + 16];
		__syncthreads();
		if(tid < 8)
			lpTmpScale0[tid] += lpTmpScale0[tid + 8];
		__syncthreads();
		if(tid < 4)
			lpTmpScale0[tid] += lpTmpScale0[tid + 4];
		__syncthreads();
		if(tid < 2)
			lpTmpScale0[tid] += lpTmpScale0[tid + 2];
		__syncthreads();
		if(tid < 1)
			lpTmpScale0[tid] += lpTmpScale0[tid + 1];
		__syncthreads();

		if(tid == 0)
			o_lpScale[neuronNum] = sqrtf(lpTmpScale0[0]);
	}

	__global__ void device_MultipyValue(F32* o_lpOutput, F32* i_lpInput0, F32* i_lpIntput1)
	{
		U32 offset = blockIdx.x * blockDim.x + threadIdx.x;

		o_lpOutput[offset] = i_lpInput0[offset] * i_lpIntput1[offset];
	}
	__global__ void device_SumSqrtf(F32* o_lpOutput, F32* i_lpInput, U32 i_inputCount)
	{
		U32 neuronNum = blockIdx.x;
		F32 sumValue = 0.0f;

		for(U32 i=0; i<i_inputCount; i++)
		{
			U32 offset = neuronNum * i_inputCount + i;

			sumValue += i_lpInput[offset];
		}

		o_lpOutput[neuronNum] = sqrt(sumValue);
	}
}



namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class WeightData_WeightNormalization_GPU : public IWeightData
	{
	private:
		thrust::device_vector<F32> lpWeight;
		thrust::device_vector<F32> lpBias;

		thrust::device_vector<F32> lpScale;			/**< neuron */
		thrust::device_vector<F32> lpVector;		/**< neuron*input */
		thrust::device_vector<F32> lpVectorScale;	/**< vector�̑傫�� neuron */

		// �덷�p
		thrust::device_vector<F32> lpDScale;
		thrust::device_vector<F32> lpDVector;

#if 0
		thrust::device_vector<F32> lpTmpValue;
#endif

		// �r���v�Z�p
		std::vector<F32> lpTmpScale_h;			/**< neuron */
		std::vector<F32> lpTmpVectorScale_h;	/**< vector�̑傫�� neuron */
		std::vector<F32> lpTmpDScale_h;

		IOptimizer* m_pOptimizer_scale;		/**< �X�J���[�̍X�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_vector;	/**< �x�N�^�[�̍X�V�p�I�v�e�B�}�C�U */
		IOptimizer* m_pOptimizer_bias;		/**< �o�C�A�X�X�V�p�I�v�e�B�}�C�U */

		U32 neuronCount;
		U32 inputCount;

		cublasHandle_t cublasHandle;

	public:
		//===========================
		// �R���X�g���N�^/�f�X�g���N�^
		//===========================
		/** �R���X�g���N�^ */
		WeightData_WeightNormalization_GPU(U32 i_neuronCount, U32 i_inputCount)
			:	lpWeight			(i_neuronCount * i_inputCount)
			,	lpBias				(i_neuronCount)
			
			,	lpScale				(i_neuronCount)
			,	lpVector			(i_neuronCount * i_inputCount)
			,	lpVectorScale		(i_neuronCount)

			,	lpDScale			(i_neuronCount)
			,	lpDVector			(i_neuronCount * i_inputCount)

#if 0
			,	lpTmpValue			(i_neuronCount * i_inputCount)
#endif

			,	lpTmpScale_h		(i_neuronCount)
			,	lpTmpVectorScale_h	(i_neuronCount)
			,	lpTmpDScale_h		(i_neuronCount)

			,	m_pOptimizer_scale	(NULL)
			,	m_pOptimizer_vector	(NULL)
			,	m_pOptimizer_bias	(NULL)

			,	neuronCount			(i_neuronCount)
			,	inputCount			(i_inputCount)
		{
			cublasCreate(&cublasHandle);
		}
		/** �f�X�g���N�^ */
		virtual ~WeightData_WeightNormalization_GPU()
		{
			if(this->m_pOptimizer_scale)
				delete this->m_pOptimizer_scale;
			if(this->m_pOptimizer_vector)
				delete this->m_pOptimizer_vector;
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			
			cublasDestroy(cublasHandle);
		}

	public:
		//===========================
		// ������
		//===========================
		ErrorCode Initialize(const wchar_t i_initializerID[], U32 i_inputCount, U32 i_outputCount)
		{
			auto& initializer = Gravisbell::Layer::NeuralNetwork::GetInitializerManager().GetInitializer(i_initializerID);

			// �d��
			thrust::host_vector<F32> lpTmpWeight(this->lpWeight.size());
			for(unsigned int weightNum=0; weightNum<lpTmpWeight.size(); weightNum++)
			{
				lpTmpWeight[weightNum] = initializer.GetParameter(i_inputCount, i_outputCount);
			}
			// �o�C�A�X
			thrust::host_vector<F32> lpTmpBias(this->lpBias.size());
			for(unsigned int biasNum=0; biasNum<lpTmpBias.size(); biasNum++)
			{
//				lpTmpBias[biasNum] = initializer.GetParameter(i_inputCount, i_outputCount);
				lpTmpBias[biasNum] = 0.0f;
			}

			// �f�o�C�X�ɃR�s�[
			thrust::device_vector<F32> lpTmpWeight_d = lpTmpWeight;
			thrust::device_vector<F32> lpTmpBias_d   = lpTmpBias;

			return this->SetData(thrust::raw_pointer_cast(&lpTmpWeight_d[0]), thrust::raw_pointer_cast(&lpTmpBias_d[0]));
		}
		S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize)
		{
			S64 readBufferByte = 0;
			
			// �X�P�[��
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpScale[0]), &i_lpBuffer[readBufferByte], this->lpScale.size() * sizeof(F32), cudaMemcpyHostToDevice);
			readBufferByte += (int)this->lpScale.size() * sizeof(F32);
			
			// �x�N�^�[
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpVector[0]), &i_lpBuffer[readBufferByte], this->lpVector.size() * sizeof(F32), cudaMemcpyHostToDevice);
			readBufferByte += (int)this->lpVector.size() * sizeof(F32);

			// �o�C�A�X
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpBias[0]), &i_lpBuffer[readBufferByte], this->lpBias.size() * sizeof(F32), cudaMemcpyHostToDevice);
			readBufferByte += (int)this->lpBias.size() * sizeof(F32);
			
#ifdef _DEBUG
			std::vector<F32> lpTmpScale(this->lpScale.size());
			std::vector<F32> lpTmpVector(this->lpVector.size());

			cudaMemcpy(&lpTmpScale[0],  thrust::raw_pointer_cast(&this->lpScale[0]),  sizeof(F32)*lpTmpScale.size(),  cudaMemcpyDeviceToHost);
			cudaMemcpy(&lpTmpVector[0], thrust::raw_pointer_cast(&this->lpVector[0]), sizeof(F32)*lpTmpVector.size(), cudaMemcpyDeviceToHost);
#endif

			// �I�v�e�B�}�C�U
			S64 useBufferSize = 0;
			// scale
			if(this->m_pOptimizer_scale)
				delete this->m_pOptimizer_scale;
			this->m_pOptimizer_scale = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// vector
			if(this->m_pOptimizer_vector)
				delete this->m_pOptimizer_vector;
			this->m_pOptimizer_vector = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;
			// bias
			if(this->m_pOptimizer_bias)
				delete this->m_pOptimizer_bias;
			this->m_pOptimizer_bias = CreateOptimizerFromBuffer_GPU(&i_lpBuffer[readBufferByte], i_bufferSize-readBufferByte, useBufferSize);
			readBufferByte += useBufferSize;

#if 1
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// �X�P�[���Z�o
				F32 sumValue = 0.0f;
				cublasSdot_v2(
					cublasHandle,
					this->inputCount,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1,
					&sumValue);

				this->lpTmpVectorScale_h[neuronNum] = sqrtf(sumValue);
			}
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpVectorScale[0]), thrust::raw_pointer_cast(&this->lpTmpVectorScale_h[0]), sizeof(F32)*this->lpVectorScale.size(), cudaMemcpyHostToDevice);
#else
			// �x�N�^�[�̃X�P�[�����Čv�Z
			{
				U32 loopCount = (this->inputCount + CALCULATE_DSCALE_BLOCK_SIZE-1) / CALCULATE_DSCALE_BLOCK_SIZE;
				device_CalculateScale<<<this->neuronCount, CALCULATE_DSCALE_BLOCK_SIZE>>>(
					thrust::raw_pointer_cast(&this->lpVectorScale[0]),
					thrust::raw_pointer_cast(&this->lpVector[0]),
					this->inputCount,
					loopCount);
			}
#endif

			// �d�݂��X�V
			this->UpdateWeight();

			return readBufferByte;
		}


		//===========================
		// �T�C�Y���擾
		//===========================
		/** Weight�̃T�C�Y���擾���� */
		U64 GetWeigthSize()const
		{
			return this->lpWeight.size();
		}
		/** Bias�̃T�C�Y���擾���� */
		U64 GetBiasSize()const
		{
			return this->lpBias.size();
		}


		//===========================
		// �l���擾
		//===========================
		/** Weight���擾���� */
		const F32* GetWeight()const
		{
			return thrust::raw_pointer_cast(&this->lpWeight[0]);
		}
		/** Bias���擾���� */
		const F32* GetBias()const
		{
			return thrust::raw_pointer_cast(&this->lpBias[0]);
		}


		//===========================
		// �l���X�V
		//===========================
		/** Weigth,Bias��ݒ肷��.
			@param	lpWeight	�ݒ肷��Weight�̒l.
			@param	lpBias		�ݒ肷��Bias�̒l. */
		ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias)
		{
			// Bias���R�s�[
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpBias[0]),   i_lpBias,   sizeof(F32)*this->lpBias.size(), cudaMemcpyDeviceToDevice);

#if 1
			// Weight���R�s�[
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpVector[0]), i_lpWeight, sizeof(F32)*this->lpWeight.size(), cudaMemcpyDeviceToDevice);

			// �X�P�[�����Z�o���āA�x�N�^�[�̃T�C�Y��1�ɂ���
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// �X�P�[���Z�o
				F32 sumValue = 0.0f;
				cublasSdot_v2(
					cublasHandle,
					this->inputCount,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1,
					&sumValue);
				F32 scale = sqrtf(sumValue);
				this->lpTmpScale_h[neuronNum] = scale;

				// �x�N�^�[�T�C�Y��1�ɂ���
				F32 alpha = 1.0f / scale;
				cublasSscal_v2(
					cublasHandle,
					this->inputCount,
					&alpha,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1);

				this->lpTmpVectorScale_h[neuronNum] = 1.0f;
			}

			// �v�Z���ʂ��R�s�[
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpScale[0]), thrust::raw_pointer_cast(&this->lpTmpScale_h[0]), sizeof(F32)*this->lpScale.size(), cudaMemcpyHostToDevice);
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpVectorScale[0]), thrust::raw_pointer_cast(&this->lpTmpVectorScale_h[0]), sizeof(F32)*this->lpVectorScale.size(), cudaMemcpyHostToDevice);
#else
			{
				U32 loopCount = (this->inputCount + CALCULATE_DSCALE_BLOCK_SIZE-1) / CALCULATE_DSCALE_BLOCK_SIZE;
				device_ResetParameter<<<this->neuronCount, CALCULATE_DSCALE_BLOCK_SIZE>>>(
					thrust::raw_pointer_cast(&this->lpScale[0]),
					thrust::raw_pointer_cast(&this->lpVector[0]),
					thrust::raw_pointer_cast(&this->lpVectorScale[0]),
					i_lpWeight,
					this->inputCount,
					loopCount);
			}
#endif

			// �d�݂��Čv�Z
			this->UpdateWeight();

			return ErrorCode::ERROR_CODE_NONE;
		}
		/** Weight,Bias���X�V����.
			@param	lpDWeight	Weight�̕ω���.
			@param	lpDBias		Bias��h�ω���. */
		ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias)
		{
#ifdef _DEBUG
			std::vector<F32> lpTmpVector(this->lpVector.size());
			std::vector<F32> lpTmpDWeight(this->lpVector.size());
			std::vector<F32> lpTmpScale(this->lpScale.size());

			cudaMemcpy(&lpTmpVector[0], thrust::raw_pointer_cast(&this->lpVector[0]), sizeof(F32)*lpTmpVector.size(), cudaMemcpyDeviceToHost);
			cudaMemcpy(&lpTmpDWeight[0], i_lpDWeight, sizeof(F32)*lpTmpDWeight.size(), cudaMemcpyDeviceToHost);
#endif

			// �덷���v�Z
			//for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			//{
			//	F32 vectorScale = this->lpTmpVectorScale_h[neuronNum];

			//	// �X�P�[���덷
			//	F32 sumValue = 0.0f;
			//	cublasSdot_v2(this->cublasHandle,
			//		this->inputCount,
			//		thrust::raw_pointer_cast(&this->lpVector[neuronNum*this->inputCount]),
			//		1,
			//		&i_lpDWeight[neuronNum*this->inputCount],
			//		1,
			//		&sumValue);
			//	this->lpTmpDScale_h[neuronNum] = sumValue / vectorScale;
			//}
			//cudaMemcpy(thrust::raw_pointer_cast(&this->lpDScale[0]), thrust::raw_pointer_cast(&this->lpTmpDScale_h[0]), sizeof(F32)*this->lpDScale.size(), cudaMemcpyHostToDevice);

			// �X�P�[���덷
			{
				U32 loopCount = (this->inputCount + CALCULATE_DSCALE_BLOCK_SIZE-1) / CALCULATE_DSCALE_BLOCK_SIZE;
				device_CalculateDScale<<<this->neuronCount, CALCULATE_DSCALE_BLOCK_SIZE>>>(
					thrust::raw_pointer_cast(&this->lpDScale[0]),
					i_lpDWeight,
					thrust::raw_pointer_cast(&this->lpVector[0]),
					thrust::raw_pointer_cast(&this->lpVectorScale[0]),
					this->inputCount,
					loopCount);
			}

			// �x�N�g���덷
			{
				device_CalculateDVector<<<this->neuronCount, this->inputCount>>>(
					thrust::raw_pointer_cast(&this->lpDVector[0]),
					thrust::raw_pointer_cast(&this->lpScale[0]),
					thrust::raw_pointer_cast(&this->lpVector[0]),
					thrust::raw_pointer_cast(&this->lpVectorScale[0]),
					i_lpDWeight,
					thrust::raw_pointer_cast(&this->lpDScale[0]));
			}

			// �덷�𔽉f
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->UpdateParameter(thrust::raw_pointer_cast(&this->lpScale[0]), thrust::raw_pointer_cast(&this->lpDScale[0]));
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->UpdateParameter(thrust::raw_pointer_cast(&this->lpVector[0]), thrust::raw_pointer_cast(&this->lpDVector[0]));
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->lpBias[0]),   i_lpDBias);

			// �X�P�[�����Čv�Z
#if 0
			for(U32 neuronNum=0; neuronNum<this->neuronCount; neuronNum++)
			{
				// �X�P�[���Z�o
				F32 sumValue = 0.0f;
				cublasSdot_v2(
					cublasHandle,
					this->inputCount,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1,
					thrust::raw_pointer_cast(&this->lpVector[neuronNum * this->inputCount]),
					1,
					&sumValue);

				this->lpTmpVectorScale_h[neuronNum] = sqrtf(sumValue);
			}
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpVectorScale[0]), thrust::raw_pointer_cast(&this->lpTmpVectorScale_h[0]), sizeof(F32)*this->lpVectorScale.size(), cudaMemcpyHostToDevice);
#elif 0
			{
				device_MultipyValue<<<this->neuronCount, this->inputCount>>>(
					thrust::raw_pointer_cast(&this->lpTmpValue[0]),
					thrust::raw_pointer_cast(&this->lpVector[0]),
					thrust::raw_pointer_cast(&this->lpVector[0]));

				device_SumSqrtf<<<this->neuronCount, 1>>>(
					thrust::raw_pointer_cast(&this->lpVectorScale[0]),
					thrust::raw_pointer_cast(&this->lpTmpValue[0]),
					this->inputCount);
			}
#else
			// �x�N�g���̃X�P�[�����v�Z
			{
				cudaThreadSynchronize();
				U32 loopCount = (this->inputCount + CALCULATE_DSCALE_BLOCK_SIZE-1) / CALCULATE_DSCALE_BLOCK_SIZE;
				device_CalculateScale<<<this->neuronCount, CALCULATE_DSCALE_BLOCK_SIZE>>>(
					thrust::raw_pointer_cast(&this->lpVectorScale[0]),
					thrust::raw_pointer_cast(&this->lpVector[0]),
					this->inputCount,
					loopCount);
				cudaThreadSynchronize();
			}
#endif
#ifdef _DEBUG
			cudaMemcpy(&lpTmpVector[0], thrust::raw_pointer_cast(&this->lpVector[0]), sizeof(F32)*lpTmpVector.size(), cudaMemcpyDeviceToHost);
			cudaMemcpy(&lpTmpScale[0],  thrust::raw_pointer_cast(&this->lpScale[0]),  sizeof(F32)*lpTmpScale.size(),  cudaMemcpyDeviceToHost);
#endif

			// �d�݂��Čv�Z
			this->UpdateWeight();

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** Weight���X�V */
		void UpdateWeight()
		{
#if 0
			device_UpdateWeight<<<this->neuronCount, this->inputCount>>>(
				thrust::raw_pointer_cast(&this->lpWeight[0]),
				thrust::raw_pointer_cast(&this->lpScale[0]),
				thrust::raw_pointer_cast(&this->lpVector[0]),
				thrust::raw_pointer_cast(&this->lpVectorScale[0]));
#else
			U32 loopCount = (this->inputCount + CALCULATE_DSCALE_BLOCK_SIZE-1) / CALCULATE_DSCALE_BLOCK_SIZE;
			device_UpdateWeight_v2<<<this->neuronCount, CALCULATE_DSCALE_BLOCK_SIZE>>>(
				thrust::raw_pointer_cast(&this->lpWeight[0]),
				thrust::raw_pointer_cast(&this->lpScale[0]),
				thrust::raw_pointer_cast(&this->lpVector[0]),
				thrust::raw_pointer_cast(&this->lpVectorScale[0]),
				this->inputCount,
				loopCount);
#endif

#ifdef _DEBUG
			std::vector<F32> lpTmpWeight(this->lpWeight.size());
			cudaMemcpy(&lpTmpWeight[0], thrust::raw_pointer_cast(&this->lpWeight[0]), sizeof(F32)*lpTmpWeight.size(), cudaMemcpyDeviceToHost);
#endif
		}

		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
		/** �I�v�e�B�}�C�U�[��ύX���� */
		ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[])
		{
			ChangeOptimizer_GPU(&this->m_pOptimizer_scale,  i_optimizerID, (U32)this->lpScale.size());
			ChangeOptimizer_GPU(&this->m_pOptimizer_vector, i_optimizerID, (U32)this->lpVector.size());
			ChangeOptimizer_GPU(&this->m_pOptimizer_bias,   i_optimizerID, (U32)this->lpBias.size());

			return ErrorCode::ERROR_CODE_NONE;
		}

		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value)
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value)
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);
		
			return ErrorCode::ERROR_CODE_NONE;
		}
		ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
		{
			if(this->m_pOptimizer_scale)
				this->m_pOptimizer_scale->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_vector)
				this->m_pOptimizer_vector->SetHyperParameter(i_parameterID, i_value);
			if(this->m_pOptimizer_bias)
				this->m_pOptimizer_bias->SetHyperParameter(i_parameterID, i_value);

			return ErrorCode::ERROR_CODE_NONE;
		}
		
		//===========================
		// ���C���[�ۑ�
		//===========================
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U64 GetUseBufferByteCount()const
		{
			U64 bufferSize = 0;

			// �{�̂̃o�C�g��
			bufferSize += sizeof(F32) * this->lpScale.size();	// �X�P�[��
			bufferSize += sizeof(F32) * this->lpVector.size();	// �x�N�^�[
			bufferSize += sizeof(F32) * this->lpBias.size();	// �o�C�A�X�W��

			// �I�v�e�B�}�C�U�[�̃o�C�g��
			bufferSize += this->m_pOptimizer_scale->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_vector->GetUseBufferByteCount();
			bufferSize += this->m_pOptimizer_bias->GetUseBufferByteCount();

			return bufferSize;
		}
		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S64 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			S64 writeBufferByte = 0;

			// �X�P�[��
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpScale[0]), this->lpScale.size() * sizeof(F32), cudaMemcpyDeviceToHost);
			writeBufferByte += (int)this->lpScale.size() * sizeof(F32);
			// �x�N�^�[
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpVector[0]), this->lpVector.size() * sizeof(F32), cudaMemcpyDeviceToHost);
			writeBufferByte += (int)this->lpVector.size() * sizeof(F32);
			// �o�C�A�X
			cudaMemcpy(&o_lpBuffer[writeBufferByte], thrust::raw_pointer_cast(&this->lpBias[0]), this->lpBias.size() * sizeof(F32), cudaMemcpyDeviceToHost);
			writeBufferByte += (int)this->lpBias.size() * sizeof(F32);

			// �I�v�e�B�}�C�U
			// scale
			writeBufferByte += this->m_pOptimizer_scale->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// vector
			writeBufferByte += this->m_pOptimizer_vector->WriteToBuffer(&o_lpBuffer[writeBufferByte]);
			// bias
			writeBufferByte += this->m_pOptimizer_bias->WriteToBuffer(&o_lpBuffer[writeBufferByte]);

			return writeBufferByte;
		}
	};

	/** �d�݃N���X���쐬����.
		�f�t�H���g.CPU����. */
	IWeightData* CreateWeightData_WeightNormalization_GPU(U32 i_neuronCount, U32 i_inputCount)
	{
		return new WeightData_WeightNormalization_GPU(i_neuronCount, i_inputCount);
	}
}
}
}