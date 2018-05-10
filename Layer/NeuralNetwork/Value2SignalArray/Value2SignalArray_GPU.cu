//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"Value2SignalArray_DATA.hpp"
#include"Value2SignalArray_FUNC.hpp"
#include"Value2SignalArray_Base.h"

#include"Value2SignalArray_GPU.cuh"
#include"Value2SignalArray_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define WORKSPACE_CODE			L"WorkSpace"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)

	__global__ void device_Value2SignalArray(
		U32 resolution,
		F32 inputMinValue,
		F32 inputMaxValue,
		const F32 lpInputBuffer[],
		F32 lpOutputBuffer[])
	{
		U32 batchNum     = blockIdx.x;
		U32 inputCh      = blockIdx.y;
		U32 inputChCount = gridDim.y;
		U32 bufferPos    = threadIdx.x;
		U32 inputChBufferSize = blockDim.x;

		U32 inputOffset = batchNum * inputChCount * inputChBufferSize + inputCh * inputChBufferSize + bufferPos;
		F32 inputValue  = lpInputBuffer[inputOffset];

		// 出力チャンネル番号をfloatで計算
		F32 fOutputCh = max(0.0f, min((F32)resolution-1 - 1e-8, (inputValue - inputMinValue) / (inputValue - inputMinValue) * resolution));

		// 整数値に変換
		U32 iOutputCh = (U32)fOutputCh;
		F32 t = fOutputCh - iOutputCh;

		U32 outputOffset0 = batchNum * (inputChCount*resolution) * inputChBufferSize + (inputCh*resolution + iOutputCh + 0) * inputChBufferSize + bufferPos;
		U32 outputOffset1 = batchNum * (inputChCount*resolution) * inputChBufferSize + (inputCh*resolution + iOutputCh + 1) * inputChBufferSize + bufferPos;

		lpOutputBuffer[outputOffset0] = (1.0f - t);
		lpOutputBuffer[outputOffset0] = t;
	}


	/** コンストラクタ */
	Value2SignalArray_GPU::Value2SignalArray_GPU(Gravisbell::GUID guid, Value2SignalArray_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Value2SignalArray_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	Value2SignalArray_GPU::~Value2SignalArray_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Value2SignalArray_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Value2SignalArray_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Value2SignalArray_LayerData_Base& Value2SignalArray_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Value2SignalArray_LayerData_Base& Value2SignalArray_GPU::GetLayerData()const
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
	ErrorCode Value2SignalArray_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// Signal -> value変換を行うための重み配列の作成
		std::vector<F32> lpSignal2ValueWeight_h(this->layerData.layerStructure.resolution);
		for(U32 i=0; i<this->layerData.layerStructure.resolution; i++)
		{
			lpSignal2ValueWeight_h[i] = (this->layerData.layerStructure.inputMaxValue - this->layerData.layerStructure.inputMinValue) * i / this->layerData.layerStructure.resolution - this->layerData.layerStructure.inputMinValue;
		}

		lpSignal2ValueWeight_d.resize(this->layerData.layerStructure.resolution);
		cudaMemcpy(thrust::raw_pointer_cast(&this->lpSignal2ValueWeight_d[0]), &lpSignal2ValueWeight_h[0], sizeof(F32)*lpSignal2ValueWeight_h.size(), cudaMemcpyHostToDevice);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Value2SignalArray_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 入力信号のチャンネルごとのバッファサイズ
		this->inputChannelSize = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		/**< 入力信号のバッチごとのバッファサイズ */
		this->inputBatchBufferSize = this->inputChannelSize * this->GetInputDataStruct().ch;

		// 一時出力バッファ(ホストメモリ)
		this->lpTmpOutputBuffer_h.resize(this->outputBufferCount * this->GetBatchSize());
		this->lpTmpBatchOutputBuffer_h.resize(this->GetBatchSize());
		for(U32 i=0; i<this->GetBatchSize(); i++)
			this->lpTmpBatchOutputBuffer_h[i] = &this->lpTmpOutputBuffer_h[this->outputBufferCount * i];

		// 処理用バッファの確保
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), WORKSPACE_CODE, sizeof(F32)*this->GetBatchSize()*this->outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Value2SignalArray_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Value2SignalArray_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 出力バッファの初期化
		cudaMemset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());
		memset(&this->lpTmpOutputBuffer_h[0], 0, sizeof(F32)*this->lpTmpOutputBuffer_h.size());

		dim3 grid(this->GetBatchSize(), this->GetInputDataStruct().ch);
		dim3 block(this->inputChannelSize);

		device_Value2SignalArray<<<grid, block>>>(
			this->layerData.layerStructure.resolution,
			this->layerData.layerStructure.inputMinValue,
			this->layerData.layerStructure.inputMaxValue,
			i_lppInputBuffer,
			o_lppOutputBuffer);

#if _DEBUG
			std::vector<F32> lpInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpInputBuffer[0], i_lppInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列の[GetOutputDataCount()]配列
		直前の計算結果を使用する */
	ErrorCode Value2SignalArray_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力バッファを入力誤差バッファにコピー
			cudaMemcpy(o_lppDInputBuffer, i_lppInputBuffer, sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);

			// 教師データ作成用の一時バッファを取得
			F32* lpSignalTeachBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE);

			// signalの教師データを作成
			cudaMemcpy(lpSignalTeachBuffer, i_lppOutputBuffer, sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			F32 alpha = 1;
			cublasSaxpy_v2(
				this->cublasHandle,
				this->outputBufferCount * this->GetBatchSize(),
				&alpha,
				i_lppDOutputBuffer,
				1,
				lpSignalTeachBuffer,
				1);

			// signal -> value変換して出力誤差バッファに格納
			alpha =  1.0f;
			F32 beta  = -1.0f;	// yには入力バッファが入っているため、教師信号-入力信号にする
			cublasSgemv_v2(
				this->cublasHandle,
				CUBLAS_OP_N,
				this->GetOutputDataStruct().ch,
				this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z,
				&alpha,
				i_lppOutputBuffer,
				this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z,
				thrust::raw_pointer_cast(&this->lpSignal2ValueWeight_d[0]),
				1,
				&beta,
				o_lppDInputBuffer,
				1);

#if _DEBUG
			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], i_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpTeachBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTeachBuffer[0], lpSignalTeachBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpDInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDInputBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// 教師データバッファの開放
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Value2SignalArray_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
