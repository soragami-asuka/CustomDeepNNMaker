//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"Residual_DATA.hpp"
#include"Residual_FUNC.hpp"
#include"Residual_Base.h"

#include"Residual_GPU.cuh"
#include"Residual_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Residual_GPU::Residual_GPU(Gravisbell::GUID guid, Residual_LayerData_GPU& i_layerData)
		:	Residual_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	m_lppInputBuffer				(NULL)			/**< 演算時の入力データ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< 入力誤差計算時の出力誤差データ */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	Residual_GPU::~Residual_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Residual_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Residual_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Residual_LayerData_Base& Residual_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Residual_LayerData_Base& Residual_GPU::GetLayerData()const
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
	ErrorCode Residual_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力差分バッファを作成
		this->lpDInputBuffer.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->lpDInputBuffer.size(); inputNum++)
		{
			this->lpDInputBuffer[inputNum].resize(this->batchSize * this->lpInputBufferCount[inputNum]);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Residual_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->lpInputBufferCount.resize(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
		{
			this->lpInputBufferCount[inputNum] = this->GetInputBufferCount(inputNum);
			if(this->lpInputBufferCount[inputNum] == 0)
				return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		}

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力バッファ保存用のバッファを作成
		this->m_lppInputBuffer.resize(this->GetInputDataCount());

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);



		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Residual_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Residual_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Residual_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer[])
	{
		// 入力バッファのアドレスを格納
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			this->m_lppInputBuffer[inputNum] = i_lpInputBuffer[inputNum];

		F32 alpha = 1.0f;

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				cublasStatus_t err = cublasSaxpy_v2(
					this->cublasHandle,
					this->lpInputBufferCount[inputNum],
					&alpha,
					&this->m_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]],
					1,
					thrust::raw_pointer_cast(&this->lpOutputBuffer[batchNum*this->outputBufferCount]),
					1);
				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

			}
		}
		cudaThreadSynchronize();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Residual_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Residual_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Residual_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBufferPrev)
	{
		// 出力誤差バッファのアドレスを配列に格納
		this->m_lppDOutputBufferPrev = i_lppDOutputBufferPrev;

		for(U32 batchNum=0; batchNum<this->batchSize; batchNum++)
		{
			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				cudaError_t err = cudaMemcpyAsync(
					thrust::raw_pointer_cast(&this->lpDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]]),
					&this->m_lppDOutputBufferPrev[batchNum*this->outputBufferCount],
					sizeof(F32) * this->lpInputBufferCount[inputNum],
					cudaMemcpyDeviceToDevice);
				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}
		}
		cudaThreadSynchronize();

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Residual_GPU::GetDInputBuffer(U32 i_dataNum)const
	{
		if(i_dataNum >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_dataNum][0]);
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Residual_GPU::GetDInputBuffer(U32 i_dataNum, BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount(i_dataNum);

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(i_dataNum), sizeof(F32)*inputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
