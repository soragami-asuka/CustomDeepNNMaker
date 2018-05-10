//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"
#define _USE_MATH_DEFINES
#include<math.h>

#include"ProbabilityArray2Value_DATA.hpp"
#include"ProbabilityArray2Value_FUNC.hpp"
#include"ProbabilityArray2Value_Base.h"

#include"ProbabilityArray2Value_GPU.cuh"
#include"ProbabilityArray2Value_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)


	__global__ void device_Value2ProbabilityArray(
		U32 outputBatchBufferSize,
		F32 outputMinValue,
		F32 outputMaxValue,
		F32 variance,
		F32 lpDInputBuffer[],
		const F32 lpOutputBuffer[],
		const F32 lpDOutputBuffer[])
	{
		U32 batchNum  = blockIdx.x;
		U32 bufferPos = threadIdx.x;
		U32 inputCh = threadIdx.y;
		U32 inputChBufferSize = blockDim.x;
		U32 inputChSize = blockDim.y;

		U32 outputOffset = outputBatchBufferSize * batchNum + bufferPos;

		F32 trueValue = lpOutputBuffer[outputOffset] + lpDOutputBuffer[outputOffset];
		F32 value = (F32)inputCh / inputChSize
				  * (outputMaxValue - outputMinValue)
				  + outputMinValue;

		U32 inputOffset = (inputChBufferSize * inputChSize * batchNum) + (inputChBufferSize * inputCh) + bufferPos;

		lpDInputBuffer[inputOffset] = 1.0f / (2.0f * (F32)M_PI * variance) * expf(-(value - trueValue)*(value - trueValue) / (2.0f * variance * variance));
	}


	/** コンストラクタ */
	ProbabilityArray2Value_GPU::ProbabilityArray2Value_GPU(Gravisbell::GUID guid, ProbabilityArray2Value_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	ProbabilityArray2Value_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	temporaryMemoryManager			(i_temporaryMemoryManager)
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	ProbabilityArray2Value_GPU::~ProbabilityArray2Value_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 ProbabilityArray2Value_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode ProbabilityArray2Value_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ProbabilityArray2Value_LayerData_Base& ProbabilityArray2Value_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ProbabilityArray2Value_LayerData_Base& ProbabilityArray2Value_GPU::GetLayerData()const
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
	ErrorCode ProbabilityArray2Value_GPU::PreProcessLearn()
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
	ErrorCode ProbabilityArray2Value_GPU::PreProcessCalculate()
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

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode ProbabilityArray2Value_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode ProbabilityArray2Value_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 出力バッファの初期化
		cudaMemset(o_lppOutputBuffer, 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());
		memset(&this->lpTmpOutputBuffer_h[0], 0, sizeof(F32)*this->lpTmpOutputBuffer_h.size());

		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 z=0; z<this->GetInputDataStruct().z; z++)
			{
				for(U32 y=0; y<this->GetInputDataStruct().y; y++)
				{
					for(U32 x=0; x<this->GetInputDataStruct().x; x++)
					{
						U32 offset = this->GetInputDataStruct().POSITION_TO_OFFSET(x, y, z, 0);

						// 最大値の番号を取得
						S32 maxPos = -1;
						cublasIsamax_v2(
							this->cublasHandle,
							this->inputBatchBufferSize,
							&i_lppInputBuffer[this->inputBatchBufferSize*batchNum + offset],
							this->inputChannelSize,
							&maxPos);

						if(maxPos <= 0)
							continue;

						this->lpTmpBatchOutputBuffer_h[batchNum][offset]
							= (F32)(maxPos - 1) / this->GetInputDataStruct().ch
							* (this->layerData.layerStructure.outputMaxValue - this->layerData.layerStructure.outputMinValue)
							+ this->layerData.layerStructure.outputMinValue;
					}
				}
			}
		}

		// CPU > GPU
		cudaMemcpy(
			o_lppOutputBuffer,
			&this->lpTmpOutputBuffer_h[0],
			sizeof(F32) * this->lpTmpOutputBuffer_h.size(),
			cudaMemcpyHostToDevice);


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
	ErrorCode ProbabilityArray2Value_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 出力バッファの初期化
			cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

#if _DEBUG
			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpOutputBuffer[0], i_lppOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			dim3 grid = this->GetBatchSize();
			dim3 block = dim3(this->inputChannelSize, this->GetInputDataStruct().ch);

			// 正規分布を計算
			device_Value2ProbabilityArray<<<grid, block>>>(
				this->outputBufferCount,
				this->layerData.layerStructure.outputMinValue,
				this->layerData.layerStructure.outputMaxValue,
				this->layerData.layerStructure.variance,
				o_lppDInputBuffer,
				i_lppOutputBuffer,
				i_lppDOutputBuffer);

#if _DEBUG
			std::vector<F32> lpTmpCalctBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTmpCalctBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// 平均化
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 bufferPos=0; bufferPos<this->inputChannelSize; bufferPos++)
				{
					// 合計を求める
					F32 sumValue = 0.0f;
					cublasSasum_v2(
						this->cublasHandle,
						this->GetInputDataStruct().ch,
						&o_lppDInputBuffer[batchNum * this->inputBufferCount + bufferPos],
						this->inputChannelSize,
						&sumValue);

					// 合計で割る
					F32 alpha = 1.0f / sumValue;
					cublasSscal_v2(
						this->cublasHandle,
						this->GetInputDataStruct().ch,
						&alpha,
						&o_lppDInputBuffer[batchNum * this->inputBufferCount + bufferPos],
						this->inputChannelSize);
				}
			}


#if _DEBUG
			std::vector<F32> lpTeachBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpTeachBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

			// 正解と出力で誤差を取る
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
	ErrorCode ProbabilityArray2Value_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
