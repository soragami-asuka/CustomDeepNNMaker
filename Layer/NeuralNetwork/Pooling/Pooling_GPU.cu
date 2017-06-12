//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"Pooling_DATA.hpp"
#include"Pooling_FUNC.hpp"
#include"Pooling_Base.h"

#include"Pooling_GPU.cuh"
#include"Pooling_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Pooling_GPU::Pooling_GPU(Gravisbell::GUID guid, Pooling_LayerData_GPU& i_layerData)
		:	Pooling_Base	(guid)
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	m_lppInputBuffer				(NULL)			/**< 演算時の入力データ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< 入力誤差計算時の出力誤差データ */
	{
        cudnnCreate(&this->cudnnHandle);
        cudnnCreateTensorDescriptor(&this->inputTensorDesc);
        cudnnCreateTensorDescriptor(&this->outputTensorDesc);
        cudnnCreatePoolingDescriptor(&this->poolingDesc);
	}
	/** デストラクタ */
	Pooling_GPU::~Pooling_GPU()
	{
        cudnnDestroyPoolingDescriptor(this->poolingDesc);
        cudnnDestroyTensorDescriptor(this->inputTensorDesc);
        cudnnDestroyTensorDescriptor(this->outputTensorDesc);
        cudnnDestroy(this->cudnnHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Pooling_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Pooling_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Pooling_LayerData_Base& Pooling_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Pooling_LayerData_Base& Pooling_GPU::GetLayerData()const
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
	ErrorCode Pooling_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Pooling_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		cudnnStatus_t err_cudnn;

		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);


		// 次元数を調べる
		S32 dataDim = 1 + 1 + 0;	// バッチ + チャンネル + 次元0
		std::vector<S32> dimInput;			// 入力データ構造
		std::vector<S32> dimInputStride;	// 入力データの各次元ごとのデータ数
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		S32 filterDim = 0;			// フィルタ次元数	入力チャンネル + 出力チャンネル + 次元
		std::vector<S32> dimFilter;
		std::vector<S32> dimStride;
		std::vector<S32> dimPadding;
		if(this->layerData.inputDataStruct.z > 1)
		{
			dataDim = 1 + 1 + 3;	// バッチ + チャンネル + 次元3

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.z;
			dimInput[3] = this->layerData.inputDataStruct.y;
			dimInput[4] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;
			
			dimOutput.resize(dataDim);
			dimOutputStride.resize(dataDim);

			filterDim = 3;	// 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.layerStructure.FilterSize.z;
			dimFilter[1] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.x;

			dimStride.resize(filterDim);
			dimStride[0] = this->layerData.layerStructure.Stride.z;
			dimStride[1] = this->layerData.layerStructure.Stride.y;
			dimStride[2] = this->layerData.layerStructure.Stride.x;

			dimPadding.resize(filterDim);
			dimPadding[0] = 0;
			dimPadding[1] = 0;
			dimPadding[2] = 0;
		}
		else if(this->layerData.inputDataStruct.y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.y;
			dimInput[3] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;

			dimOutput.resize(dataDim);
			dimOutputStride.resize(dataDim);

			filterDim = 2;	// 次元2

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[1] = this->layerData.layerStructure.FilterSize.x;

			dimStride.resize(filterDim);
			dimStride[0] = this->layerData.layerStructure.Stride.y;
			dimStride[1] = this->layerData.layerStructure.Stride.x;
			
			dimPadding.resize(filterDim);
			dimPadding[0] = 0;
			dimPadding[1] = 0;
		}
		else if(this->layerData.inputDataStruct.x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;


			filterDim = 1;	// 次元1

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.layerStructure.FilterSize.x;

			dimStride.resize(filterDim);
			dimStride[0] = this->layerData.layerStructure.Stride.x;
			
			dimPadding.resize(filterDim);
			dimPadding[0] = 0;
		}
		else
		{
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}
		dimOutput.resize(dataDim);
		dimOutputStride.resize(dataDim);


		// CUDNNの入力データ構造設定
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->inputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimInputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNNのプーリング設定
		err_cudnn = cudnnSetPoolingNdDescriptor(
			this->poolingDesc,
			CUDNN_POOLING_MAX,
			CUDNN_PROPAGATE_NAN,
			filterDim,
			&dimFilter[0],
			&dimPadding[0],
			&dimStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 出力データ構造を取得
		err_cudnn = cudnnGetPoolingNdForwardOutputDim(
			this->poolingDesc,
			this->inputTensorDesc,
			dataDim,
			&dimOutput[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// CUDNNの出力データ構造とGravisbellの出力データ構造が一致することを確認
		Gravisbell::Vector3D<S32> outputVector;
		S32 outputBatchSize = dimOutput[0];
		S32 outputCh = dimOutput[1];
		if(dataDim == 5)
		{
			outputVector.z = dimOutput[2];
			outputVector.y = dimOutput[3];
			outputVector.x = dimOutput[4];
		}
		else if(dataDim == 4)
		{
			outputVector.z = 1;
			outputVector.y = dimOutput[2];
			outputVector.x = dimOutput[3];
		}
		else if(dataDim == 3)
		{
			outputVector.z = 1;
			outputVector.y = 1;
			outputVector.x = dimOutput[2];
		}
		if(outputBatchSize != this->batchSize)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputCh != this->GetOutputDataStruct().ch)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.z != this->GetOutputDataStruct().z)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.y != this->GetOutputDataStruct().y)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.x != this->GetOutputDataStruct().x)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// CUDNNのデータ構造設定
		for(S32 i=0; i<dataDim; i++)
		{
			dimOutputStride[i] = 1;
			for(S32 j=i+1; j<dataDim; j++)
				dimOutputStride[i] *= dimOutput[j];
		}
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Pooling_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Pooling_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Pooling_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファのアドレスを格納
		this->m_lppInputBuffer = i_lpInputBuffer;

		// プーリング処理を実行
		F32 alpha = 1.0f;
		F32 beta  = 0.0f;
		cudnnStatus_t err_cudnn = cudnnPoolingForward(
			this->cudnnHandle,
			this->poolingDesc,
			&alpha,
			this->inputTensorDesc,
			i_lpInputBuffer,
			&beta,
			this->outputTensorDesc,
			thrust::raw_pointer_cast(&this->lpOutputBuffer[0]));
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Pooling_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Pooling_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
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
	ErrorCode Pooling_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 出力誤差バッファのアドレスを配列に格納
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのアドレスを格納
			this->m_lpDInputBuffer_d = o_lppDInputBuffer;

			F32 alpha = 1.0f;
			F32 beta  = 0.0f;
			cudnnStatus_t err_cudnn = cudnnPoolingBackward(
				this->cudnnHandle,
				this->poolingDesc,
				&alpha,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				this->outputTensorDesc,
				this->m_lppDOutputBufferPrev,
				this->inputTensorDesc,
				this->m_lppInputBuffer,
				&beta,
				this->inputTensorDesc,
				this->m_lpDInputBuffer_d);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Pooling_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(o_lppDInputBuffer, i_lppDOutputBuffer);
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Pooling_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Pooling_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
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
