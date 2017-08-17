//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
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

	/** コンストラクタ */
	GaussianNoise_GPU::GaussianNoise_GPU(Gravisbell::GUID guid, GaussianNoise_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	GaussianNoise_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	m_lppInputBuffer				(NULL)			/**< 演算時の入力データ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< 入力誤差計算時の出力誤差データ */
	{
	}
	/** デストラクタ */
	GaussianNoise_GPU::~GaussianNoise_GPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 GaussianNoise_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode GaussianNoise_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	GaussianNoise_LayerData_Base& GaussianNoise_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const GaussianNoise_LayerData_Base& GaussianNoise_GPU::GetLayerData()const
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
	ErrorCode GaussianNoise_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode GaussianNoise_GPU::PreProcessCalculate()
	{
		cudnnStatus_t err_cudnn;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}
	


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode GaussianNoise_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode GaussianNoise_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを格納
		this->m_lppInputBuffer = i_lpInputBuffer;

		// 入力バッファを出力バッファにコピー
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), i_lpInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);

		F32 average  = this->layerData.layerStructure.Average  + this->GetRuntimeParameterByStructure().GaussianNoise_Bias;
		F32 variance = this->layerData.layerStructure.Variance * this->GetRuntimeParameterByStructure().GaussianNoise_Power;

		// ノイズを加算
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


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GaussianNoise_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
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
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode GaussianNoise_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファのアドレスを配列に格納
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode GaussianNoise_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER GaussianNoise_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
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
