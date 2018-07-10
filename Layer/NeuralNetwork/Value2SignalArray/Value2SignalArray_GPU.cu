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
	
#define THREAD_PER_BLOCK	32

	__global__ void device_Value2SignalArray(
		const F32 i_lpValue[],
		F32 o_lpSignalArray[],
		U32 i_resolution,
		F32 i_minValue,
		F32 i_maxValue,
		U32 i_loopCount,
		U32 i_bufferPerCh)
	{
		U32 batchNum     = blockIdx.x;
		U32 inputCh      = blockIdx.y;
		U32 inputChCount = gridDim.y;
		U32 tid          = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = loopNum * THREAD_PER_BLOCK + tid;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 inputOffset = batchNum * inputChCount * i_bufferPerCh + inputCh * i_bufferPerCh + bufferPos;
			F32 inputValue  = i_lpValue[inputOffset];

			// 出力チャンネル番号をfloatで計算
			U32 outputCh = max(0, min(i_resolution-1, (U32)((i_resolution-1) * (inputValue - i_minValue) / (i_maxValue - i_minValue) + 0.5f) ));
			
			U32 outputOffset = batchNum * (inputChCount*i_resolution) * i_bufferPerCh + (inputCh*i_resolution + outputCh) * i_bufferPerCh + bufferPos;
			o_lpSignalArray[outputOffset] = 1.0f;

			// 整数値に変換
			//U32 iOutputCh = (U32)fOutputCh;
			//F32 t = fOutputCh - iOutputCh;

			//U32 outputOffset0 = batchNum * (inputChCount*resolution) * i_inputChBufferSize + (inputCh*resolution + iOutputCh + 0) * i_inputChBufferSize + bufferPos;
			//U32 outputOffset1 = batchNum * (inputChCount*resolution) * i_inputChBufferSize + (inputCh*resolution + iOutputCh + 1) * i_inputChBufferSize + bufferPos;

			//lpOutputBuffer[outputOffset0] = (1.0f - t);
			//lpOutputBuffer[outputOffset1] = t;
		}
	}


	/** 信号配列を値に変換する.
		<inputChNo, batchSize> <32>
		*/
	__global__ void device_SignalArray2Value(
		F32* o_lpTeach,
		const F32* i_lpOutput,
		const F32* i_lpDOutput,
		U32 i_resolution, U32 i_bufferPerCh, U32 i_loopCount,
		F32 i_minValue,
		F32 i_maxValue)
	{
		U32 batchNum = blockIdx.y;
		U32 inputChNo = blockIdx.x;
		U32 inputChCount = gridDim.x;
		U32 tid = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = loopNum * THREAD_PER_BLOCK + tid;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 inputOffset = (batchNum * inputChCount + inputChNo) * i_bufferPerCh + bufferPos;

			// 最大値を求める
			U32 maxNum = 0;
			F32 maxValue = -FLT_MAX;
			for(U32 outputNum=0; outputNum<i_resolution; outputNum++)
			{
				U32 outputChNum = inputChNo * i_resolution + outputNum;
				U32 outputOffset = (batchNum * (inputChCount * i_resolution) + outputChNum) * i_bufferPerCh + bufferPos;

				F32 value = i_lpOutput[outputOffset] + i_lpDOutput[outputOffset];

				if(value > maxValue)
				{
					maxNum = outputNum;
					maxValue = value;
				}
			}

			o_lpTeach[inputOffset] = ((F32)maxNum / (i_resolution-1)) * (i_maxValue - i_minValue) + i_minValue;
		}
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
		for(U32 i=0; i<(U32)this->layerData.layerStructure.resolution; i++)
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
		this->bufferPerChannel = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		/**< 入力信号のバッチごとのバッファサイズ */
		this->inputBatchBufferSize = this->bufferPerChannel * this->GetInputDataStruct().ch;

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

		dim3 grid(this->GetBatchSize(), this->GetInputDataStruct().ch);
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->bufferPerChannel + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		device_Value2SignalArray<<<grid, block>>>(
			i_lppInputBuffer,
			o_lppOutputBuffer,
			this->layerData.layerStructure.resolution,
			this->layerData.layerStructure.inputMinValue,
			this->layerData.layerStructure.inputMaxValue,
			loopCount,
			this->bufferPerChannel);

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
#if _DEBUG
			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], i_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
			
			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			dim3 grid(this->GetInputDataStruct().ch, this->GetBatchSize());
			dim3 block(THREAD_PER_BLOCK);
			U32 loopCount = (this->bufferPerChannel + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

			device_SignalArray2Value<<<grid, block>>>(
				o_lppDInputBuffer,
				i_lppOutputBuffer,
				i_lppDOutputBuffer,
				this->layerData.layerStructure.resolution,
				this->bufferPerChannel,
				loopCount,
				this->layerData.layerStructure.inputMinValue, this->layerData.layerStructure.inputMaxValue);

#if _DEBUG
			std::vector<F32> lpDOutputBuffer2(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer2[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpTeachBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTeachBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// 正解と入力で誤差を取る
			F32 alpha = -1;
			cublasSaxpy_v2(
				this->cublasHandle,
				this->inputBufferCount * this->GetBatchSize(),
				&alpha,
				i_lppInputBuffer,
				1,
				o_lppDInputBuffer,
				1);

#if _DEBUG
			std::vector<F32> lpDInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDInputBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

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
