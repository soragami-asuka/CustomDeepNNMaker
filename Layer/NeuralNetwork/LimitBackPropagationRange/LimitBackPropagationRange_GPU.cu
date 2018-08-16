//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"LimitBackPropagationRange_DATA.hpp"
#include"LimitBackPropagationRange_FUNC.hpp"
#include"LimitBackPropagationRange_Base.h"

#include"LimitBackPropagationRange_GPU.cuh"
#include"LimitBackPropagationRange_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
#define THREAD_PER_BLOCK	32
#define CALC_BATCH_MAX	(256)
#define CALC_INPUT_MAX	(1024)
	
	__global__ void device_ReLimitBackPropagationRange(
		U32 chCount,
		U32 startX, U32 startY, U32 startZ,
		U32 inputXCount,  U32 inputYCount,  U32 inputZCount,
		U32 outputXCount, U32 outputYCount, U32 outputZCount,
		U32 i_bufferPerCh,
		U32 i_loopCount,
		const F32 lpDOutputBuffer[],
		F32 lpDInputBuffer[])
	{
		U32 chNum    = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 tid = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 x = bufferPos % inputXCount;
			U32 y = bufferPos / inputXCount % inputYCount;
			U32 z = bufferPos / inputXCount / inputYCount % inputZCount;

			if(x < startX || x >= startX + outputXCount)
				continue;
			if(y < startY || y >= startY + outputYCount)
				continue;
			if(z < startZ || z >= startZ + outputZCount)
				continue;

			U32 offset = batchNum * (chCount * i_bufferPerCh) + chNum * i_bufferPerCh + bufferPos;

			lpDInputBuffer[offset] = lpDOutputBuffer[offset];
		}
	}

	/** コンストラクタ */
	LimitBackPropagationRange_GPU::LimitBackPropagationRange_GPU(Gravisbell::GUID guid, LimitBackPropagationRange_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	LimitBackPropagationRange_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	LimitBackPropagationRange_GPU::~LimitBackPropagationRange_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 LimitBackPropagationRange_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode LimitBackPropagationRange_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	LimitBackPropagationRange_LayerData_Base& LimitBackPropagationRange_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const LimitBackPropagationRange_LayerData_Base& LimitBackPropagationRange_GPU::GetLayerData()const
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
	ErrorCode LimitBackPropagationRange_GPU::PreProcessLearn()
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
	ErrorCode LimitBackPropagationRange_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// CHあたりのバッファ数
		this->bufferCountPerCh = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		return ErrorCode::ERROR_CODE_NONE;
	}



	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode LimitBackPropagationRange_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode LimitBackPropagationRange_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファを出力バッファにコピー
		cudaMemcpy(o_lppOutputBuffer, i_lppInputBuffer, sizeof(F32)*this->GetBatchSize()*this->outputBufferCount, cudaMemcpyDeviceToDevice);

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
	ErrorCode LimitBackPropagationRange_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 出力バッファの初期化
			cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());
			
			// 計算
			dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
			dim3 block(THREAD_PER_BLOCK);
			U32 loopCount = (this->bufferCountPerCh + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

			device_ReLimitBackPropagationRange<<<grid, block>>>(
				this->GetOutputDataStruct().ch,
				this->layerData.layerStructure.startPosition.x, this->layerData.layerStructure.startPosition.y, this->layerData.layerStructure.startPosition.z,
				this->GetInputDataStruct().x, this->GetInputDataStruct().y, this->GetInputDataStruct().z,
				this->layerData.layerStructure.boxSize.x, this->layerData.layerStructure.boxSize.y, this->layerData.layerStructure.boxSize.z,
				this->bufferCountPerCh, loopCount,
				i_lppDOutputBuffer,
				o_lppDInputBuffer);
			
#if _DEBUG
			std::vector<F32> lpDInputBuffer(this->inputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDInputBuffer[0], o_lppDInputBuffer, sizeof(F32) * this->inputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);

			std::vector<F32> lpDOutputBuffer(this->outputBufferCount * this->GetBatchSize());
			cudaMemcpy(&lpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(F32) * this->outputBufferCount * this->GetBatchSize(), cudaMemcpyDeviceToHost);
#endif

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode LimitBackPropagationRange_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
