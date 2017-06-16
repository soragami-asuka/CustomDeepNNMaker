//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// GPU処理用
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
	/** ベクトルの要素同士の掛け算. */
	__global__ void cuda_func_multiplVector(const F32* i_lpInputBufferA, const F32* i_lpInputBufferB, F32* o_lpOutputBuffer, U32 i_bufferSize)
	{
		const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(bufferPos >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
			return;

		o_lpOutputBuffer[bufferPos] = i_lpInputBufferA[bufferPos] * i_lpInputBufferB[bufferPos];
	}
	/** ベクトルの要素同士の掛け算. */
	__global__ void cuda_func_multiplVectorWithScaler(const F32* i_lpInputBufferA, const F32* i_lpInputBufferB, F32* o_lpOutputBuffer, U32 i_bufferSize, F32 alpha, F32 beta)
	{
		const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(bufferPos >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
			return;

		o_lpOutputBuffer[bufferPos] = alpha * i_lpInputBufferA[bufferPos] * i_lpInputBufferB[bufferPos] + beta * o_lpOutputBuffer[bufferPos];
	}
}

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	FullyConnect_GPU::FullyConnect_GPU(Gravisbell::GUID guid, FullyConnect_LayerData_GPU& i_layerData)
		:	FullyConnect_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	FullyConnect_GPU::~FullyConnect_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 FullyConnect_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FullyConnect_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	FullyConnect_LayerData_Base& FullyConnect_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_LayerData_Base& FullyConnect_GPU::GetLayerData()const
	{
		return this->layerData;
	}


	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode FullyConnect_GPU::PreProcessLearn(U32 batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// バイアス更新用のベクトルを作成
		lpBiasUpdateVector_d.resize(this->batchSize);
		{
			thrust::host_vector<F32> lpBuf(this->batchSize, 1.0f);
			this->lpBiasUpdateVector_d = lpBuf;
		}

		// パラメータ変化量
		this->lpDBias.resize(this->neuronCount);
		this->lpDNeuron.resize(this->neuronCount * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_GPU::PreProcessCalculate(U32 batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->layerData.lppNeuron_d.size() != this->neuronCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファを作成
		this->lpOutputBuffer_d.resize(this->batchSize * this->outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
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
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FullyConnect_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファを保管
		this->m_lppInputBuffer_d = i_lpInputBuffer;

		// バイアスを出力信号にコピーする
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

		// ニューロンT×入力信号
		{
			// C = aAB + bC;

			F32 alpha = 1.0f;
			F32 beta  = 1.0f;	// バイアスがCにコピー済みなのでそのまま利用するために1.0を指定

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				this->neuronCount,	// 行列Aの行数
				this->batchSize,	// 行列Bの列数
				this->inputBufferCount,	// 行列Aの列数,行列Bの行数
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// 行列A
				this->inputBufferCount,										// 行列Aの転置前の行数
				i_lpInputBuffer,											// 行列B
				this->inputBufferCount,										// 行列Bの転置前の行数
				&beta,
				thrust::raw_pointer_cast(&lpOutputBuffer_d[0]),
				this->outputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer_d[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
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
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FullyConnect_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファのアドレスを格納
		this->m_lppDOutputBuffer_d = i_lppDOutputBuffer;
		// 入力誤差バッファのアドレスを配列に格納
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;

		// 入力誤差差分を計算
		if(this->m_lpDInputBuffer_d)
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->inputBufferCount,	// 行列Aの行数
				this->batchSize,		// 行列Bの列数
				this->neuronCount,		// 行列Aの列数,行列Bの行数
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// 行列A
				this->inputBufferCount,										// 行列Aの転置前の行数
				this->m_lppDOutputBuffer_d,									// 行列B
				this->neuronCount,											// 行列Bの転置前の行数
				&beta,
				this->m_lpDInputBuffer_d,
				this->inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FullyConnect_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		Gravisbell::ErrorCode errCode = this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		
		// バイアス変化量計算
		{
			F32 alpha = 1.0f;
			F32 beta  = 0;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->neuronCount,		// 行列Aの行数
				1,						// 行列Bの列数
				this->batchSize,		// 行列Aの列数,行列Bの行数
				&alpha,
				this->m_lppDOutputBuffer_d,	// 行列A
				this->neuronCount,											// 行列Aの転置前の行数
				thrust::raw_pointer_cast(&this->lpBiasUpdateVector_d[0]),	// 行列B
				this->batchSize,											// 行列Bの転置前の行数
				&beta,
				thrust::raw_pointer_cast(&this->lpDBias[0]),
				this->neuronCount);
		}

		// ニューロン変化量計算
		{
			// ニューロンの誤差を計算して加算する
			{
				F32 alpha = 1.0f;
				F32 beta  = 0;

				cublasSgemm(
					this->cublasHandle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					this->inputBufferCount,	// 行列Aの行数
					this->neuronCount,		// 行列Bの列数
					this->batchSize,		// 行列Aの列数,行列Bの行数
					&alpha,
					this->m_lppInputBuffer_d,		// 行列A
					this->inputBufferCount,										// 行列Aの転置前の行数
					this->m_lppDOutputBuffer_d,	// 行列B
					this->neuronCount,										// 行列Bの転置前の行数
					&beta,
					thrust::raw_pointer_cast(&this->lpDNeuron[0]),
					this->inputBufferCount);
			}
		}


		// 誤差を反映
		if(this->m_pOptimizer_bias)
			this->m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),   thrust::raw_pointer_cast(&this->lpDBias[0]));
		if(this->m_pOptimizer_neuron)
			this->m_pOptimizer_neuron->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]), thrust::raw_pointer_cast(&this->lpDNeuron[0]));


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER FullyConnect_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
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
