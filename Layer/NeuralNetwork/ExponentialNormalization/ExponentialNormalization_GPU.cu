//======================================
// バッチ正規化レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include<algorithm>

#include"ExponentialNormalization_DATA.hpp"
#include"ExponentialNormalization_FUNC.hpp"
#include"ExponentialNormalization_Base.h"

#include"ExponentialNormalization_GPU.cuh"
#include"ExponentialNormalization_LayerData_GPU.cuh"

#define WORKSPACE_CODE			L"WorkSpace"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
	
#define THREAD_PER_BLOCK	32

	/** 出力を計算する
		dim = <ch,1,1>
		block = <32,1,1>
	*/
	__global__ void device_CalculateOutput(F32* o_lpOutput, const F32* i_lpInput,  const F32* i_lpAverage, const F32* i_lpVariance, U32 i_inputCountPerChannel, U32 i_loopCount, F32 i_epsilon)
	{
		U32 ch = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 chCount = gridDim.x;

		U32 tid = threadIdx.x;

		F32 average = i_lpAverage[ch];
		F32 deviation = sqrtf(i_lpVariance[ch] + i_epsilon);

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_inputCountPerChannel)
				continue;

			U32 offset = (batchNum * chCount + ch) * i_inputCountPerChannel + bufferPos;

			F32 value = i_lpInput[offset];

			o_lpOutput[offset] = (value - average) / deviation;
		}
	}

	/** 出力を計算する
		dim = <ch,1,1>
		block = <32,1,1>
	*/
	__global__ void device_CalculateDInput(F32* o_lpDInput, const F32* i_lpDOutput, const F32* i_lpVariance, U32 i_inputCountPerChannel, U32 i_loopCount, F32 i_epsilon)
	{
		U32 ch = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 chCount = gridDim.x;

		U32 tid = threadIdx.x;

		F32 deviation = sqrtf(i_lpVariance[ch] + i_epsilon);

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_inputCountPerChannel)
				continue;

			U32 offset = (batchNum * chCount + ch) * i_inputCountPerChannel + bufferPos;

			F32 value = i_lpDOutput[offset];

			o_lpDInput[offset] = value / deviation;
		}
	}

	/** CHごとの平均を求めて更新する
		dim = <ch,1,1>
		block = <32,1,1>
	*/
	__global__ void device_UpdateChAverage(F32* o_lpAverage, const F32* i_lpInputValue, U32 i_inputCountPerChannel, U32 i_batchSize, U32 i_loopCount, F32 i_alpha)
	{
		__shared__ F32 lpTmpSumValue[THREAD_PER_BLOCK];

		U32 chNum = blockIdx.x;
		U32 chCount = gridDim.x;
		U32 tid = threadIdx.x;
		U32 inputCount = chCount * i_inputCountPerChannel;

		// DWeightとVectorの乗算を計算
		lpTmpSumValue[tid] = 0.0f;
		for(U32 batchNum=0; batchNum<i_batchSize; batchNum++)
		{
			for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
			{
				U32 inputNum = THREAD_PER_BLOCK * loopNum + tid;
				if(inputNum >= i_inputCountPerChannel)
					continue;

				U32 offset = batchNum * inputCount + chNum * i_inputCountPerChannel + inputNum;

				lpTmpSumValue[tid] += i_lpInputValue[offset];
			}
		}
		__syncthreads();

		// 合計
		if(tid < 16)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 16];
		__syncthreads();
		if(tid < 8)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 8];
		__syncthreads();
		if(tid < 4)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 4];
		__syncthreads();
		if(tid < 2)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 2];
		__syncthreads();
		if(tid < 1)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 1];
		__syncthreads();

		if(tid == 0)
		{
			F32 average = lpTmpSumValue[tid] / (i_inputCountPerChannel * i_batchSize);

			o_lpAverage[chNum] = i_alpha * average + (1.0f - i_alpha) * o_lpAverage[chNum];
		}
	}

	/** CHごとの分散を求めて更新する
		dim = <ch,1,1>
		block = <32,1,1>
	*/
	__global__ void device_UpdateChVariance(F32* o_lpVariance, const F32* i_lpAverage, const F32* i_lpInputValue, U32 i_inputCountPerChannel, U32 i_batchSize, U32 i_loopCount, F32 i_alpha)
	{
		__shared__ F32 lpTmpSumValue[THREAD_PER_BLOCK];

		U32 chNum = blockIdx.x;
		U32 chCount = gridDim.x;
		U32 tid = threadIdx.x;
		U32 inputCount = chCount * i_inputCountPerChannel;
		F32 average = i_lpAverage[chNum];

		// DWeightとVectorの乗算を計算
		lpTmpSumValue[tid] = 0.0f;
		for(U32 batchNum=0; batchNum<i_batchSize; batchNum++)
		{
			for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
			{
				U32 inputNum = THREAD_PER_BLOCK * loopNum + tid;
				if(inputNum >= i_inputCountPerChannel)
					continue;

				U32 offset = batchNum * inputCount + chNum * i_inputCountPerChannel + inputNum;

				lpTmpSumValue[tid] += (i_lpInputValue[offset] - average) * (i_lpInputValue[offset] - average);
			}
		}
		__syncthreads();

		// 合計
		if(tid < 16)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 16];
		__syncthreads();
		if(tid < 8)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 8];
		__syncthreads();
		if(tid < 4)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 4];
		__syncthreads();
		if(tid < 2)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 2];
		__syncthreads();
		if(tid < 1)
			lpTmpSumValue[tid] += lpTmpSumValue[tid + 1];
		__syncthreads();

		if(tid == 0)
		{
			F32 variance = lpTmpSumValue[tid] / (i_inputCountPerChannel * i_batchSize);

			o_lpVariance[chNum] = i_alpha * variance + (1.0f - i_alpha) * o_lpVariance[chNum];
		}
	}

	/** コンストラクタ */
	ExponentialNormalization_GPU::ExponentialNormalization_GPU(Gravisbell::GUID guid, ExponentialNormalization_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	ExponentialNormalization_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)				/**< 入力バッファ数 */
		,	outputBufferCount		(0)				/**< 出力バッファ数 */
		,	channeclBufferCount		(0)				/**< 1チャンネル当たりのバッファ数 */
		,	temporaryMemoryManager	(i_temporaryMemoryManager)
	{
	}
	/** デストラクタ */
	ExponentialNormalization_GPU::~ExponentialNormalization_GPU()
	{
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 ExponentialNormalization_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode ExponentialNormalization_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ExponentialNormalization_LayerData_Base& ExponentialNormalization_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ExponentialNormalization_LayerData_Base& ExponentialNormalization_GPU::GetLayerData()const
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
	ErrorCode ExponentialNormalization_GPU::PreProcessLearn()
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
	ErrorCode ExponentialNormalization_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// チャンネルごとのバッファ数を確認
		this->channeclBufferCount = this->GetInputDataStruct().z * this->GetInputDataStruct().y * this->GetInputDataStruct().x;
		if(this->channeclBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode ExponentialNormalization_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode ExponentialNormalization_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 計算
		dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->channeclBufferCount + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		device_CalculateOutput<<<grid,block>>>(
			o_lppOutputBuffer,
			i_lppInputBuffer,
			thrust::raw_pointer_cast(&this->layerData.lpMean[0]),
			thrust::raw_pointer_cast(&this->layerData.lpVariance[0]),
			this->channeclBufferCount,
			loopCount,
			this->layerData.layerStructure.epsilon);
		
#ifdef _DEBUG
			std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
			cudaMemcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

			std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
			cudaMemcpy(&lpTmpOutputBuffer[0], o_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

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
	ErrorCode ExponentialNormalization_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// 計算
			dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
			dim3 block(THREAD_PER_BLOCK);
			U32 loopCount = (this->channeclBufferCount + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

			device_CalculateDInput<<<grid,block>>>(
				o_lppDInputBuffer,
				i_lppDOutputBuffer,
				thrust::raw_pointer_cast(&this->layerData.lpVariance[0]),
				this->channeclBufferCount,
				loopCount,
				this->layerData.layerStructure.epsilon);


#ifdef _DEBUG
			std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
			cudaMemcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

			std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
			cudaMemcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

			std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
			cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

			std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
			cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode ExponentialNormalization_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		}

		// 学習回数を更新
		this->layerData.learnTime++;

		// CHごとに平均と分散を求める
		F32 alpha = 0.0f;
		if(this->layerData.learnTime < this->layerData.layerStructure.InitParameterTime)
			alpha = 1.0f / (this->layerData.learnTime + 1);
		else
			alpha = std::min<F32>(1.0f, this->GetRuntimeParameterByStructure().AccelCoeff * 2 / (this->layerData.layerStructure.ExponentialTime + 1));

		dim3 grid(this->GetOutputDataStruct().ch);
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->channeclBufferCount + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		// 平均を更新する
		device_UpdateChAverage<<<grid,block>>>(
			thrust::raw_pointer_cast(&this->layerData.lpMean[0]),
			i_lppInputBuffer,
			this->channeclBufferCount,
			this->GetBatchSize(),
			loopCount,
			alpha);

		// 分散を更新する
		device_UpdateChVariance<<<grid,block>>>(
			thrust::raw_pointer_cast(&this->layerData.lpVariance[0]),
			thrust::raw_pointer_cast(&this->layerData.lpMean[0]),
			i_lppInputBuffer,
			this->channeclBufferCount,
			this->GetBatchSize(),
			loopCount,
			alpha);


#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], i_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpMean(this->layerData.lpMean.size());
		cudaMemcpy(&lpMean[0], thrust::raw_pointer_cast(&this->layerData.lpMean[0]), sizeof(float)*lpMean.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpVariance(this->layerData.lpVariance.size());
		cudaMemcpy(&lpVariance[0], thrust::raw_pointer_cast(&this->layerData.lpVariance[0]), sizeof(float)*lpVariance.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
