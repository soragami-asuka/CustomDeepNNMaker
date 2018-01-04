//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include"FullyConnect_DATA.hpp"
#include"FullyConnect_FUNC.hpp"
#include"FullyConnect_Base.h"

#include"FullyConnect_GPU.cuh"
#include"FullyConnect_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define BLOCK_SIZE	(16)

namespace
{
	/** ベクトルの要素同士の掛け算. */
	__global__ void cuda_func_multiplVector(const F32* i_lpInputBufferA, const F32* i_lpInputBufferB, F32* o_lpOutputBuffer, U32 i_bufferSize)
	{
		const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(bufferPos >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
			return;

		o_lpOutputBuffer[bufferPos] = i_lpInputBufferA[bufferPos] * i_lpInputBufferB[bufferPos];
	}
	/** ベクトルの要素同士の掛け算. */
	__global__ void cuda_func_multiplVectorWithScaler(const F32* i_lpInputBufferA, const F32* i_lpInputBufferB, F32* o_lpOutputBuffer, U32 i_bufferSize, F32 alpha, F32 beta)
	{
		const U32 bufferPos = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		if(bufferPos >= i_bufferSize)	// 分岐するが末尾のwarpだけなので、処理速度に影響はないはず...
			return;

		o_lpOutputBuffer[bufferPos] = alpha * i_lpInputBufferA[bufferPos] * i_lpInputBufferB[bufferPos] + beta * o_lpOutputBuffer[bufferPos];
	}
}

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	FullyConnect_GPU::FullyConnect_GPU(Gravisbell::GUID guid, FullyConnect_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FullyConnect_Base				(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	neuronCount						(0)		/**< ニューロン数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
	{
		cublasCreate(&cublasHandle);
	}
	/** デストラクタ */
	FullyConnect_GPU::~FullyConnect_GPU()
	{
		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 FullyConnect_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode FullyConnect_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	FullyConnect_LayerData_Base& FullyConnect_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const FullyConnect_LayerData_Base& FullyConnect_GPU::GetLayerData()const
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
	ErrorCode FullyConnect_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// バイアス更新用のベクトルを作成
		lpBiasUpdateVector_d.resize(this->GetBatchSize());
		{
			thrust::host_vector<F32> lpBuf(this->GetBatchSize(), 1.0f);
			this->lpBiasUpdateVector_d = lpBuf;
		}

		// パラメータ変化量
		this->lpDBias.resize(this->neuronCount);
		this->lpDNeuron.resize(this->neuronCount * this->inputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->GetNeuronCount();
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// ニューロンバッファのサイズ確認
		if(this->layerData.lppNeuron_d.size() != this->neuronCount * this->inputBufferCount)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FullyConnect_GPU::PreProcessLoop()
	{
		return ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FullyConnect_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			// ※とりあえずCPU側で処理.
			// 基本的に1回しか通らないから処理負荷に影響は与えない・・・はず
			// 超手抜き


			U32 PROCTIME_MAX = 5;			// 実行最大値
			F32	VARIANCE_TOLERANCE = 0.1f;	// 分散交差(許容範囲)

			std::vector<F32> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);

			U32 procTime = 0;
			do
			{
				// 演算を実行
				ErrorCode err = this->CalculateBase(i_lppInputBuffer, o_lppOutputBuffer);
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				// バッファをコピー
				cudaMemcpy(&lpTmpOutputBuffer[0], &o_lppOutputBuffer[0], sizeof(F32)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

				// 出力の分散を求める
				F32 variance = 0.0f;
				F32 average  = 0.0f;
				{
					// 平均を求める
					for(U32 outputNum=0; outputNum<lpTmpOutputBuffer.size(); outputNum++)
					{
						average += lpTmpOutputBuffer[outputNum];
					}
					average /= lpTmpOutputBuffer.size();

					// 分散を求める
					for(U32 outputNum=0; outputNum<lpTmpOutputBuffer.size(); outputNum++)
					{
						variance += (lpTmpOutputBuffer[outputNum] - average) * (lpTmpOutputBuffer[outputNum] - average);
					}
					variance /= lpTmpOutputBuffer.size();
				}

				if( abs(variance - 1.0f) < VARIANCE_TOLERANCE)
					break;

				// 標準偏差で重みを割って更新する
				F32 deviation = sqrtf(variance);
				{
					thrust::host_vector<F32> lpTmpNeuron = this->layerData.lppNeuron_d;
					thrust::host_vector<F32> lpTmpBias   = this->layerData.lpBias_d;

					for(U32 neuronNum=0; neuronNum<lpTmpNeuron.size(); neuronNum++)
					{
						lpTmpNeuron[neuronNum] /= deviation;
					}
					for(U32 neuronNum=0; neuronNum<lpTmpBias.size(); neuronNum++)
					{
						lpTmpBias[neuronNum] /= deviation;
					}

					this->layerData.lppNeuron_d = lpTmpNeuron;
					this->layerData.lpBias_d    = lpTmpBias;
				}

				procTime++;
			}while(procTime < 5);
		}
		else
		{
			ErrorCode err = this->CalculateBase(i_lppInputBuffer, o_lppOutputBuffer);
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode FullyConnect_GPU::CalculateBase(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// バイアスを出力信号にコピーする
		{
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				cudaError_t err = cudaMemcpy(
					&o_lppOutputBuffer[batchNum * this->outputBufferCount],
					thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),
					sizeof(F32) * this->neuronCount,
					cudaMemcpyDeviceToDevice);
				if(err != 0)
					return ERROR_CODE_CUDA_COPY_MEMORY;
			}
		}

		// ニューロンT×入力信号
		{
			// C = aAB + bC;

			F32 alpha = 1.0f;
			F32 beta  = 1.0f;	// バイアスがCにコピー済みなのでそのまま利用するために1.0を指定

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_T,
				CUBLAS_OP_N,
				this->neuronCount,	// 行列Aの行数
				this->GetBatchSize(),	// 行列Bの列数
				this->inputBufferCount,	// 行列Aの列数,行列Bの行数
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// 行列A
				this->inputBufferCount,										// 行列Aの転置前の行数
				i_lppInputBuffer,											// 行列B
				this->inputBufferCount,										// 行列Bの転置前の行数
				&beta,
				&o_lppOutputBuffer[0],
				this->outputBufferCount);
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
	ErrorCode FullyConnect_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差差分を計算
		if(o_lppDInputBuffer)
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->inputBufferCount,	// 行列Aの行数
				this->GetBatchSize(),		// 行列Bの列数
				this->neuronCount,		// 行列Aの列数,行列Bの行数
				&alpha,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),	// 行列A
				this->inputBufferCount,										// 行列Aの転置前の行数
				i_lppDOutputBuffer,											// 行列B
				this->neuronCount,											// 行列Bの転置前の行数
				&beta,
				o_lppDInputBuffer,
				this->inputBufferCount);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FullyConnect_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		std::vector<F32> lpDOutputBuffer_h(this->outputBufferCount * this->GetBatchSize());
		cudaMemcpy(&lpDOutputBuffer_h[0], i_lppDOutputBuffer, sizeof(F32)*lpDOutputBuffer_h.size(), cudaMemcpyDeviceToHost);

		std::vector<F32> lpInputBuffer_h(this->inputBufferCount * this->GetBatchSize());
		cudaMemcpy(&lpInputBuffer_h[0], i_lppInputBuffer, sizeof(F32)*lpInputBuffer_h.size(), cudaMemcpyDeviceToHost);


		// 入力誤差計算
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;

		
		// バイアス変化量計算
		{
			F32 alpha = 1.0f;
			F32 beta  = 0;

			cublasSgemm(
				this->cublasHandle,
				CUBLAS_OP_N,
				CUBLAS_OP_N,
				this->neuronCount,		// 行列Aの行数
				1,						// 行列Bの列数
				this->GetBatchSize(),		// 行列Aの列数,行列Bの行数
				&alpha,
				i_lppDOutputBuffer,		// 行列A
				this->neuronCount,											// 行列Aの転置前の行数
				thrust::raw_pointer_cast(&this->lpBiasUpdateVector_d[0]),	// 行列B
				this->GetBatchSize(),										// 行列Bの転置前の行数
				&beta,
				thrust::raw_pointer_cast(&this->lpDBias[0]),
				this->neuronCount);
		}

		// ニューロン変化量計算
		{
			// ニューロンの誤差を計算して加算する
			{
				F32 alpha = 1.0f;
				F32 beta  = 0;

				cublasSgemm(
					this->cublasHandle,
					CUBLAS_OP_N,
					CUBLAS_OP_T,
					this->inputBufferCount,	// 行列Aの行数
					this->neuronCount,		// 行列Bの列数
					this->GetBatchSize(),		// 行列Aの列数,行列Bの行数
					&alpha,
					i_lppInputBuffer,		// 行列A
					this->inputBufferCount,										// 行列Aの転置前の行数
					i_lppDOutputBuffer,	// 行列B
					this->neuronCount,										// 行列Bの転置前の行数
					&beta,
					thrust::raw_pointer_cast(&this->lpDNeuron[0]),
					this->inputBufferCount);
			}
		}


		// 誤差を反映
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),   thrust::raw_pointer_cast(&this->lpDBias[0]));
		if(this->layerData.m_pOptimizer_neuron)
			this->layerData.m_pOptimizer_neuron->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]), thrust::raw_pointer_cast(&this->lpDNeuron[0]));

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
