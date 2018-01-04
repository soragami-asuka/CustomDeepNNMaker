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
	Dropout_GPU::Dropout_GPU(Gravisbell::GUID guid, Dropout_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Dropout_Base					(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
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
	ErrorCode Dropout_GPU::PreProcessLearn()
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
	ErrorCode Dropout_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 出力バッファを作成
		{
			int n = this->GetBatchSize();
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


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Dropout_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Dropout_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetRuntimeParameterByStructure().UseDropOut)
		{
			cudnnStatus_t err = cudnnDropoutForward(
				this->cudnnHandle,
				this->dropoutDesc,
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->outputTensorDesc,
				o_lppOutputBuffer,
				this->m_pReserve,
				this->reserveSize);

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}
		else
		{
			cudaMemcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
		}

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
	ErrorCode Dropout_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			if(this->GetRuntimeParameterByStructure().UseDropOut)
			{
				cudnnStatus_t err = cudnnDropoutBackward(
					this->cudnnHandle,
					this->dropoutDesc,
					this->outputTensorDesc,
					i_lppDOutputBuffer,
					this->inputTensorDesc,
					o_lppDInputBuffer,
					this->m_pReserve,
					this->reserveSize);

				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}
			else
			{
				cudaMemcpy(o_lppDInputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Dropout_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}




} // Gravisbell;
} // Layer;
} // NeuralNetwork;
