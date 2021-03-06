//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"GlobalAveragePooling_DATA.hpp"
#include"GlobalAveragePooling_FUNC.hpp"
#include"GlobalAveragePooling_Base.h"

#include"GlobalAveragePooling_GPU.cuh"
#include"GlobalAveragePooling_LayerData_GPU.cuh"

#include<device_functions.hpp>

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(32)

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	namespace
	{
		// 初回用
		__global__ void cuda_func_average(const F32* i_lpInputBuffer, F32* o_lpOutputBuffer, const U32 i_inputChSize, U32 i_outputChSize)
		{
			const U32 batchNo = blockIdx.z;
			const U32 chNo    = blockIdx.y;
			const U32 chCount = gridDim.y;

			const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;

			const U32 outputPos = batchNo * (i_outputChSize * chCount) + chNo * i_outputChSize + blockIdx.x;
			const U32 inputPos  = batchNo * (i_inputChSize  * chCount) + chNo * i_inputChSize  + bufferPos;

			__shared__ F32 lpTmpBuf[BLOCK_SIZE*2];
			if(bufferPos >= i_inputChSize)
				lpTmpBuf[threadIdx.x]  = 0.0f;
			else
				lpTmpBuf[threadIdx.x]  = i_lpInputBuffer[inputPos];
			__syncthreads();

			if(threadIdx.x < 16)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 16];
			__syncthreads();
			if(threadIdx.x < 8)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 8];
			__syncthreads();
			if(threadIdx.x < 4)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 4];
			__syncthreads();
			if(threadIdx.x < 2)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 2];
			__syncthreads();
			if(threadIdx.x < 1)
				lpTmpBuf[threadIdx.x] += lpTmpBuf[threadIdx.x + 1];
			__syncthreads();

			if(threadIdx.x < 1)
				o_lpOutputBuffer[outputPos] = lpTmpBuf[0];
		}


		// 出力誤差を入力誤差に変換する
		__global__ void cuda_func_DOutput_to_DInput(const F32* i_lpDOutputBuffer, F32* o_lpDInputBuffer, const U32 i_inputChSize)
		{
			const U32 batchNo = blockIdx.z;
			const U32 chNo    = blockIdx.y;
			const U32 chCount = gridDim.y;

			const U32 inpuBufferPos   = blockIdx.x * BLOCK_SIZE + threadIdx.x;
			
			const U32 inputPos  = batchNo * (chCount * i_inputChSize) + chNo * i_inputChSize + inpuBufferPos;
			const U32 outputPos = batchNo *  chCount + chNo;


			if(inpuBufferPos < i_inputChSize)
			{
				o_lpDInputBuffer[inputPos] = i_lpDOutputBuffer[outputPos] / i_inputChSize;
			}
		}
	}


	/** コンストラクタ */
	GlobalAveragePooling_GPU::GlobalAveragePooling_GPU(Gravisbell::GUID guid, GlobalAveragePooling_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	GlobalAveragePooling_Base		(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)				/**< 入力バッファ数 */
		,	outputBufferCount				(0)				/**< 出力バッファ数 */
		,	cublasHandle					(NULL)
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	GlobalAveragePooling_GPU::~GlobalAveragePooling_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 GlobalAveragePooling_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode GlobalAveragePooling_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	GlobalAveragePooling_LayerData_Base& GlobalAveragePooling_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const GlobalAveragePooling_LayerData_Base& GlobalAveragePooling_GPU::GetLayerData()const
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
	ErrorCode GlobalAveragePooling_GPU::PreProcessLearn()
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
	ErrorCode GlobalAveragePooling_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 1CHあたりのサイズを計算
		this->chSize = this->GetInputDataStruct().x * this->GetInputDataStruct().y * this->GetInputDataStruct().z;

		// 一時バッファの確保
		this->lpTmpBuffer0.resize((this->chSize + 31)/32*32 * this->GetInputDataStruct().ch * this->GetBatchSize(), 0.0f);
		this->lpTmpBuffer1.resize((this->chSize + 31)/32*32 * this->GetInputDataStruct().ch * this->GetBatchSize(), 0.0f);
		this->lpTmpOutputBuffer_host.resize(this->outputBufferCount * this->GetBatchSize());

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode GlobalAveragePooling_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode GlobalAveragePooling_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{

		// 初回処理
		U32 tmpInputBufferCount = this->chSize;
		U32 tmpOutputBufferCount = (tmpInputBufferCount + (BLOCK_SIZE-1))/BLOCK_SIZE;
		{
			dim3 grid(tmpOutputBufferCount, this->GetInputDataStruct().ch, this->GetBatchSize());

			cuda_func_average<<<grid, BLOCK_SIZE>>>(i_lppInputBuffer, thrust::raw_pointer_cast(&this->lpTmpBuffer0[0]), tmpInputBufferCount, tmpOutputBufferCount);
		}
		thrust::device_vector<F32>* pTmpBufferIn  = &this->lpTmpBuffer0;
		thrust::device_vector<F32>* pTmpBufferOut = &this->lpTmpBuffer1;


		while(tmpOutputBufferCount > 1)
		{
			tmpInputBufferCount = tmpOutputBufferCount;
			tmpOutputBufferCount = (tmpInputBufferCount + (BLOCK_SIZE-1))/BLOCK_SIZE;

			dim3 grid(tmpOutputBufferCount, this->GetInputDataStruct().ch, this->GetBatchSize());

			cuda_func_average<<<grid, BLOCK_SIZE>>>(
				thrust::raw_pointer_cast(&(*pTmpBufferIn)[0]),
				thrust::raw_pointer_cast(&(*pTmpBufferOut)[0]),
				tmpInputBufferCount, tmpOutputBufferCount);

			thrust::device_vector<F32>* pTmpBufferTmp = pTmpBufferIn;
			pTmpBufferIn  = pTmpBufferOut;
			pTmpBufferOut = pTmpBufferTmp;
		}

		// 各CHの要素をchサイズで除算して本体に格納
		cudaMemcpy(
			thrust::raw_pointer_cast(&this->lpTmpOutputBuffer_host[0]),
			thrust::raw_pointer_cast(&(*pTmpBufferIn)[0]),
			sizeof(F32)*this->outputBufferCount*this->GetBatchSize(),
			cudaMemcpyDeviceToHost);
		for(U32 outputNum=0; outputNum<this->outputBufferCount*this->GetBatchSize(); outputNum++)
		{
			lpTmpOutputBuffer_host[outputNum] /= this->chSize;
		}
		cudaMemcpy(o_lppOutputBuffer, &lpTmpOutputBuffer_host[0], sizeof(F32)*this->outputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);

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
	ErrorCode GlobalAveragePooling_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファを0クリア
			cudaMemset(o_lppDInputBuffer, 0, sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

			// ch数で割った値を代入
			{
				dim3 grid((this->chSize + (BLOCK_SIZE-1))/BLOCK_SIZE, this->GetInputDataStruct().ch, this->GetBatchSize());

				cuda_func_DOutput_to_DInput<<<grid, BLOCK_SIZE>>>(
					i_lppDOutputBuffer,
					o_lppDInputBuffer,
					this->chSize);
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode GlobalAveragePooling_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
