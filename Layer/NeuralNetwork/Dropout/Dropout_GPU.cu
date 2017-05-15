//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
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


	/** コンストラクタ */
	Dropout_GPU::Dropout_GPU(Gravisbell::GUID guid, Dropout_LayerData_GPU& i_layerData)
		:	Dropout_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	onLearning						(false)			/**< 学習処理中フラグ */
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
	/** デストラクタ */
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
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Dropout_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Dropout_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Dropout_LayerData_Base& Dropout_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Dropout_LayerData_Base& Dropout_GPU::GetLayerData()const
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
	ErrorCode Dropout_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 学習処理中フラグを設定する
		this->onLearning = true;

		// 入力差分バッファを作成
		this->lpDInputBuffer_d.resize(this->batchSize * this->inputBufferCount);
		
		// 出力バッファを作成
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

		// ドロップアウト設定を作成
		{
			// 乱数ジェネレータ用メモリを確保
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
			
			// ドロップアウトバッファ用メモリを確保
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

			// ドロップアウト設定を作成
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


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Dropout_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 学習処理中フラグを降ろす
		this->onLearning = false;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Dropout_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Dropout_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Dropout_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを配列に格納
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


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
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
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
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
	// 学習処理
	//================================
	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Dropout_GPU::Training(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		// 出力誤差バッファのアドレスを配列に格納
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


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
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
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
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
