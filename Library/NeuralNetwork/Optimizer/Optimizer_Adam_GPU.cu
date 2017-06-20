//===============================================
// 最適化ルーチン(Adam)
//===============================================
#include"stdafx.h"

#include<vector>

#include"Optimizer_Adam_base.h"

// CUDA用
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda.h>
#include <cublas_v2.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "device_launch_parameters.h"
#pragma warning(pop)

#define BLOCK_SIZE	(32)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	namespace
	{
		/** ベクトルの要素同士の掛け算. */
		__global__ void cuda_func_updateParameter(F32* io_lpParameter, const F32* i_lpDParameter, const U32 i_bufferSize, F32* io_lpParameterM, F32* io_lpParameterV, F32 i_alpha, F32 i_beta1, F32 i_beta2, F32 i_epsilon, F32 i_beta1Pows, F32 i_beta2Pows)
		{
			const U32 paramNum = blockIdx.x * BLOCK_SIZE + threadIdx.x;
			if(paramNum >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
				return;

			io_lpParameterM[paramNum] = i_beta1 * io_lpParameterM[paramNum] + (1.0f - i_beta1) * i_lpDParameter[paramNum];
			io_lpParameterV[paramNum] = i_beta2 * io_lpParameterV[paramNum] + (1.0f - i_beta2) * i_lpDParameter[paramNum] * i_lpDParameter[paramNum];

			F32 tmpM = io_lpParameterM[paramNum] / (1.0f - i_beta1Pows);
			F32 tmpV = io_lpParameterV[paramNum] / (1.0f - i_beta2Pows);

			io_lpParameter[paramNum] += i_alpha * (tmpM / (sqrt(tmpV) + i_epsilon));
		}
	}

	class Optimizer_Adam_GPU : public Optimizer_Adam_base
	{
	public:
		thrust::device_vector<F32> lpParameterM;
		thrust::device_vector<F32> lpParameterV;

		F32 m_beta1Pows;	/**< β1の階乗値 */
		F32 m_beta2Pows;	/**< β2の階乗値 */

	public:
		/** コンストラクタ */
		Optimizer_Adam_GPU(U32 i_parameterCount)
			:	Optimizer_Adam_base	(i_parameterCount)
			,	m_beta2Pows	(1.0f)	/**< β2の階乗値 */
			,	m_beta1Pows	(1.0f)	/**< β1の階乗値 */
		{
			this->lpParameterM.resize(this->m_parameterCount);
			this->lpParameterV.resize(this->m_parameterCount);
		}
		/** デストラクタ */
		virtual ~Optimizer_Adam_GPU()
		{
		}

	public:
		//===========================
		// 処理
		//===========================
		/** パラメータを更新する.
			@param io_lpParamter	更新するパラメータ.
			@param io_lpDParameter	パラメータの変化量. */
		ErrorCode UpdateParameter(F32 io_lpParameter[], const F32 i_lpDParameter[])
		{
			this->m_beta1Pows *= this->m_beta1;
			this->m_beta2Pows *= this->m_beta2;
			
			dim3 grid((this->m_parameterCount +(BLOCK_SIZE - 1))/BLOCK_SIZE , 1, 1);
			dim3 block(BLOCK_SIZE, 1, 1);

			cuda_func_updateParameter<<<grid, block>>>(
				io_lpParameter,
				i_lpDParameter,
				this->m_parameterCount,
				thrust::raw_pointer_cast(&this->lpParameterM[0]),
				thrust::raw_pointer_cast(&this->lpParameterV[0]),
				this->m_alpha, this->m_beta1, this->m_beta2, this->m_epsilon,
				this->m_beta1Pows, this->m_beta2Pows);


			return ErrorCode::ERROR_CODE_NONE;
		}
		
	public:
		//===========================
		// 保存
		//===========================
		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const
		{
			U32 writePos = WriteToBufferBase(o_lpBuffer);

			// M
			cudaMemcpy(&o_lpBuffer[writePos], thrust::raw_pointer_cast(&this->lpParameterM[0]), sizeof(F32)*this->m_parameterCount, cudaMemcpyDeviceToHost);
			writePos += sizeof(F32)*this->m_parameterCount;
			// V
			cudaMemcpy(&o_lpBuffer[writePos], thrust::raw_pointer_cast(&this->lpParameterV[0]), sizeof(F32)*this->m_parameterCount, cudaMemcpyDeviceToHost);
			writePos += sizeof(F32)*this->m_parameterCount;

			// beta1^n
			memcpy(&o_lpBuffer[writePos], &this->m_beta1Pows, sizeof(F32));
			writePos += sizeof(F32);
			// beta2^n
			memcpy(&o_lpBuffer[writePos], &this->m_beta2Pows, sizeof(F32));
			writePos += sizeof(F32);

			return writePos;
		}
	};

	/** オプティマイザを作成する */
	Optimizer_Adam_base* CreateOptimizer_Adam_GPU(U32 i_parameterCount)
	{
		return new Optimizer_Adam_GPU(i_parameterCount);
	}
	/** オプティマイザをバッファから作成する */
	IOptimizer* CreateOptimizerFromBuffer_Adam_GPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize)
	{
		Optimizer_Adam_base* pOptimizer = CreateOptimizerFromBuffer_Adam(i_lpBuffer, i_bufferSize, o_useBufferSize, CreateOptimizer_Adam_GPU);
		if(pOptimizer == NULL)
			return NULL;
		Optimizer_Adam_GPU* pOptimizerGPU = dynamic_cast<Optimizer_Adam_GPU*>(pOptimizer);
		if(pOptimizerGPU == NULL)
		{
			delete pOptimizer;
			return NULL;
		}

		// M
		cudaMemcpy(thrust::raw_pointer_cast(&pOptimizerGPU->lpParameterM[0]), &i_lpBuffer[o_useBufferSize], sizeof(F32)*pOptimizerGPU->lpParameterM.size(), cudaMemcpyHostToDevice);
		o_useBufferSize += sizeof(F32)*pOptimizerGPU->lpParameterM.size();
		// V
		cudaMemcpy(thrust::raw_pointer_cast(&pOptimizerGPU->lpParameterV[0]), &i_lpBuffer[o_useBufferSize], sizeof(F32)*pOptimizerGPU->lpParameterV.size(), cudaMemcpyHostToDevice);
		o_useBufferSize += sizeof(F32)*pOptimizerGPU->lpParameterV.size();

		// beta1^n
		memcpy(&pOptimizerGPU->m_beta1Pows, &i_lpBuffer[o_useBufferSize], sizeof(F32));
		o_useBufferSize += sizeof(F32);
		// beta2^n
		memcpy(&pOptimizerGPU->m_beta2Pows, &i_lpBuffer[o_useBufferSize], sizeof(F32));
		o_useBufferSize += sizeof(F32);

		return pOptimizer;
	}
	/** オプティマイザーを更新する.異なる型だった場合は強制的に指定の型に変換される. */
	ErrorCode ChangeOptimizer_Adam_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount)
	{
		Optimizer_Adam_base* pOptimizer = dynamic_cast<Optimizer_Adam_base*>(*io_ppOptimizer);
		if(pOptimizer == NULL)
		{
			if(*io_ppOptimizer)
				delete *io_ppOptimizer;

			*io_ppOptimizer = CreateOptimizer_Adam_GPU(i_parameterCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell