//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"MergeAdd_DATA.hpp"
#include"MergeAdd_FUNC.hpp"
#include"MergeAdd_Base.h"

#include"MergeAdd_GPU.cuh"
#include"MergeAdd_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

#define THREAD_PER_BLOCK	32

	/** 入力を足し合わせる.
		<outputChCount, batchSize> <32>
		@param	o_lpOutput			出力バッファ
		@param	i_outputChCount		出力バッファのCH数
		@param	i_inputLyaerCount	入力レイヤー数
		@param	i_lppInput			入力バッファ
		@param	i_lpInputChCount	入力バッファのCH数
		@param	i_bufferPerCh		チャンネルあたりのバッファ数
		@param	i_loopCount			1スレッドあたりの実行ループ回数
		*/
	__global__ void device_SumInput(F32* o_lpOutput, U32 i_outputChCount, U32 i_inputLayerCount, const F32*const* i_lppInput, const U32* i_lpInputChCount, U32 i_bufferPerCh, U32 i_loopCount, F32 i_scale)
	{
		U32 chNum    = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 tid = threadIdx.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 outputOffset = (batchNum * i_outputChCount + chNum) * i_bufferPerCh + bufferPos;

			// 出力初期化
			o_lpOutput[outputOffset] = 0.0f;
			for(U32 inputLayerNum=0; inputLayerNum<i_inputLayerCount; inputLayerNum++)
			{
				if(chNum >= i_lpInputChCount[inputLayerNum])
					continue;

				U32 inputOffset = (batchNum * i_lpInputChCount[inputLayerNum] + chNum) *i_bufferPerCh + bufferPos;

				o_lpOutput[outputOffset] += i_lppInput[inputLayerNum][inputOffset];
			}

			// スケールを掛ける
			o_lpOutput[outputOffset] *= i_scale;
		}
	}

	/** 入力誤差を計算する.
		<outputChCount, batchSize> <32>
		@param	o_lppDInput			入力誤差バッファ
		@param	i_lpInputChCount	入力バッファのCH数
		@param	i_inputLyaerCount	入力レイヤー数
		@param	i_lpDOutput			出力誤差バッファ
		@param	i_bufferPerCh		チャンネルあたりのバッファ数
		@param	i_loopCount			1スレッドあたりの実行ループ回数
		*/
	__global__ void device_CalculateDInput(F32** o_lppDInput, const U32* i_lpInputChCount, U32 i_inputLayerCount, const F32* i_lpDOutput, U32 i_bufferPerCh, U32 i_loopCount, F32 i_scale)
	{
		U32 chNum    = blockIdx.x;
		U32 batchNum = blockIdx.y;
		U32 tid = threadIdx.x;
		U32 outputChCount = gridDim.x;

		for(U32 loopNum=0; loopNum<i_loopCount; loopNum++)
		{
			U32 bufferPos = tid*i_loopCount + loopNum;
			if(bufferPos >= i_bufferPerCh)
				continue;

			U32 outputOffset = (batchNum * outputChCount + chNum) * i_bufferPerCh + bufferPos;

			// 入力誤差計算
			for(U32 inputLayerNum=0; inputLayerNum<i_inputLayerCount; inputLayerNum++)
			{
				if(chNum >= i_lpInputChCount[inputLayerNum])
					continue;

				U32 inputOffset = (batchNum * i_lpInputChCount[inputLayerNum] + chNum) *i_bufferPerCh + bufferPos;

				o_lppDInput[inputLayerNum][inputOffset] = i_lpDOutput[outputOffset] * i_scale;
			}
		}
	}


	/** コンストラクタ */
	MergeAdd_GPU::MergeAdd_GPU(Gravisbell::GUID guid, MergeAdd_LayerData_GPU& i_layerData, const std::vector<IODataStruct>& i_lpInputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	MergeAdd_Base					(guid, i_lpInputDataStruct, i_layerData.GetOutputDataStruct(&i_lpInputDataStruct[0], (U32)i_lpInputDataStruct.size()))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	MergeAdd_GPU::~MergeAdd_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 MergeAdd_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode MergeAdd_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	MergeAdd_LayerData_Base& MergeAdd_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const MergeAdd_LayerData_Base& MergeAdd_GPU::GetLayerData()const
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
	ErrorCode MergeAdd_GPU::PreProcessLearn()
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
	ErrorCode MergeAdd_GPU::PreProcessCalculate()
	{
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

		// CHあたりのバッファ数
		this->bufferCountPerCh = this->GetOutputDataStruct().x * this->GetOutputDataStruct().y * this->GetOutputDataStruct().z;

		// 各入力レイヤーのCH数
		thrust::host_vector<U32> lpInputChCount(this->GetInputDataCount());
		for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
		{
			lpInputChCount[inputNum] = this->GetInputDataStruct(inputNum).ch;
		}
		this->lpInputChCount_d = lpInputChCount;

		// 入力信号の先頭アドレスの配列
		// バッファの確保のみ
		this->lppInputBuffer_d.resize(this->GetInputDataCount());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode MergeAdd_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 演算処理
	//================================
	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode MergeAdd_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
#if 0
		// 出力バッファを初期化
		cudaMemset(&o_lppOutputBuffer[0], 0, sizeof(F32)*this->outputBufferCount*this->GetBatchSize());

		F32 alpha = this->layerData.layerStructure.Scale;
		for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
		{
			for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
			{
				cublasStatus_t err = cublasSaxpy_v2(
					this->cublasHandle,
					min(this->lpInputBufferCount[inputNum], outputBufferCount),
					&alpha,
					&i_lppInputBuffer[inputNum][batchNum * this->lpInputBufferCount[inputNum]],
					1,
					&o_lppOutputBuffer[batchNum*this->outputBufferCount],
					1);
				if(err != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

			}
		}
		cudaThreadSynchronize();
#else
		// 入力信号配列をDeviceにコピー
		cudaMemcpy(thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]), i_lppInputBuffer, sizeof(F32*)*this->lppInputBuffer_d.size(), cudaMemcpyHostToDevice);

		// 計算
		dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->bufferCountPerCh + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		device_SumInput<<<grid,block>>>(
			o_lppOutputBuffer, this->GetOutputDataStruct().ch,
			this->GetInputDataCount(), thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]), thrust::raw_pointer_cast(&this->lpInputChCount_d[0]),
			this->bufferCountPerCh,
			loopCount,
			this->layerData.layerStructure.Scale);
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
	ErrorCode MergeAdd_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
#if 0
			// 入力誤差バッファの初期化
			for(U32 inputNum=0; inputNum<this->GetInputDataCount(); inputNum++)
			{
				cudaMemset(o_lppDInputBuffer[inputNum], 0, sizeof(F32)*this->lpInputBufferCount[inputNum]*this->GetBatchSize());
			}
			
			F32 alpha = this->layerData.layerStructure.Scale;
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				for(U32 inputNum=0; inputNum<this->lpInputBufferCount.size(); inputNum++)
				{
					cublasStatus_t err = cublasSaxpy_v2(
						this->cublasHandle,
						min(this->lpInputBufferCount[inputNum], outputBufferCount),
						&alpha,
						&i_lppDOutputBuffer[batchNum*this->outputBufferCount],
						1,
						&o_lppDInputBuffer[inputNum][batchNum*this->lpInputBufferCount[inputNum]],
						1);

					if(err != 0)
						return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
				}
			}
			cudaThreadSynchronize();
#else
		// 入力誤差信号配列をDeviceにコピー
		cudaMemcpy(thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]), o_lppDInputBuffer, sizeof(F32*)*this->lppInputBuffer_d.size(), cudaMemcpyHostToDevice);

		// 計算
		dim3 grid(this->GetOutputDataStruct().ch, this->GetBatchSize());
		dim3 block(THREAD_PER_BLOCK);
		U32 loopCount = (this->bufferCountPerCh + (THREAD_PER_BLOCK-1)) / THREAD_PER_BLOCK;

		device_CalculateDInput<<<grid, block>>>(
			thrust::raw_pointer_cast(&this->lppInputBuffer_d[0]),
			thrust::raw_pointer_cast(&this->lpInputChCount_d[0]), this->GetInputDataCount(),
			i_lppDOutputBuffer,
			this->bufferCountPerCh,
			loopCount,
			this->layerData.layerStructure.Scale);
#endif
		}


#ifdef _DEBUG
		std::vector<std::vector<float>> lpTmpInputBuffer(this->GetInputDataCount());
		for(int i=0; i<lpTmpInputBuffer.size(); i++)
		{
			lpTmpInputBuffer[i].resize(this->GetBatchSize() * this->lpInputBufferCount[i]);
			cudaMemcpy(&lpTmpInputBuffer[i][0], i_lppInputBuffer[i], sizeof(float)*lpTmpInputBuffer[i].size(), cudaMemcpyDeviceToHost);
		}

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], i_lppOutputBuffer, sizeof(float)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<std::vector<float>> lpTmpDInputBuffer(this->GetInputDataCount());
		for(int i=0; i<lpTmpInputBuffer.size(); i++)
		{
			lpTmpDInputBuffer[i].resize(this->GetBatchSize() * this->lpInputBufferCount[i]);
			cudaMemcpy(&lpTmpDInputBuffer[i][0], o_lppDInputBuffer[i], sizeof(float)*lpTmpDInputBuffer[i].size(), cudaMemcpyDeviceToHost);
		}
#endif


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode MergeAdd_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[], BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}



} // Gravisbell;
} // Layer;
} // NeuralNetwork;
