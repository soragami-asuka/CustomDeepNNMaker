//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// CPU処理用
//======================================
#include"stdafx.h"

#include"Activation_Discriminator_DATA.hpp"
#include"Activation_Discriminator_FUNC.hpp"
#include"Activation_Discriminator_Base.h"

#include"Activation_Discriminator_GPU.cuh"
#include"Activation_Discriminator_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Activation_Discriminator_GPU::Activation_Discriminator_GPU(Gravisbell::GUID guid, Activation_Discriminator_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Activation_Discriminator_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData						(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount				(0)		/**< 入力バッファ数 */
		,	outputBufferCount				(0)		/**< 出力バッファ数 */
		,	cudnnHandle		(NULL)
		,	inputTensorDesc	(NULL)
		,	outputTensorDesc	(NULL)
	{
		cublasCreate(&cublasHandle);
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateTensorDescriptor(&tmpOutputTensorDesc);
	}
	/** デストラクタ */
	Activation_Discriminator_GPU::~Activation_Discriminator_GPU()
	{
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(tmpOutputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
		if(cublasHandle)		cublasDestroy(cublasHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Activation_Discriminator_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Activation_Discriminator_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	ILayerData& Activation_Discriminator_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const ILayerData& Activation_Discriminator_GPU::GetLayerData()const
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
	ErrorCode Activation_Discriminator_GPU::PreProcessLearn()
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
	ErrorCode Activation_Discriminator_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		// 一時出力バッファを作成
		this->lpTmpOutputBuffer_d.resize(this->GetBatchSize() * this->inputBufferCount);
		this->lpDInputBuffer_h.resize(this->GetBatchSize() * this->inputBufferCount);
		{
			int n = this->GetBatchSize();
			int c = this->GetInputDataStruct().ch;
			int h = this->GetInputDataStruct().z * this->GetInputDataStruct().y;
			int w = this->GetInputDataStruct().x;

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

			err = cudnnSetTensorNdDescriptor(this->tmpOutputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

			// 入力バッファも同じサイズなのでディスクリプタを作っておく
			err = cudnnSetTensorNdDescriptor(this->inputTensorDesc,
				CUDNN_DATA_FLOAT,
				4,
				dimA,
				strideA );

			if(err != 0)
				return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}

		// 出力バッファを作成
		this->lpDOutputBuffer_h.resize(this->GetBatchSize() * this->outputBufferCount);	/**< 出力誤差バッファのCPU側アドレス */
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
		}


		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Activation_Discriminator_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Activation_Discriminator_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		F32 alpha = 1.0f;
		F32 beta = 0.0f;
		cudnnStatus_t err =	cudnnSoftmaxForward(
				this->cudnnHandle,
				CUDNN_SOFTMAX_ACCURATE,
				CUDNN_SOFTMAX_MODE_INSTANCE,
				&alpha,
				this->inputTensorDesc,
				i_lppInputBuffer,
				&beta,
				this->tmpOutputTensorDesc,
				thrust::raw_pointer_cast(&this->lpTmpOutputBuffer_d[0]));
		if(err != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

		cublasStatus_t err_cublas =	cublasScopy_v2(this->cublasHandle,
			this->GetBatchSize(),
			thrust::raw_pointer_cast(&this->lpTmpOutputBuffer_d[0]),
			this->inputBufferCount,
			o_lppOutputBuffer,
			1);
		if(err_cublas != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;


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
	ErrorCode Activation_Discriminator_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 出力誤差をホスト側にコピー
			cudaMemcpy(thrust::raw_pointer_cast(&this->lpDOutputBuffer_h[0]), i_lppDOutputBuffer, sizeof(F32)*this->lpDOutputBuffer_h.size(), cudaMemcpyDeviceToHost);

			// 入力誤差を計算
			for(U32 batchNum=0; batchNum<this->GetBatchSize(); batchNum++)
			{
				//this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 0] = (       this->lpOutputBuffer_d[batchNum]) *  this->lpDOutputBuffer_h[batchNum];
				//this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 1] = (1.0f - this->lpOutputBuffer_d[batchNum]) * -this->lpDOutputBuffer_h[batchNum];
				this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 0] =  this->lpDOutputBuffer_h[batchNum];
				this->lpDInputBuffer_h[batchNum*this->inputBufferCount + 1] = -this->lpDOutputBuffer_h[batchNum];
			}

			cudaMemcpy(o_lppDInputBuffer, thrust::raw_pointer_cast(&this->lpDInputBuffer_h[0]), sizeof(F32)*this->inputBufferCount*this->GetBatchSize(), cudaMemcpyHostToDevice);
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Activation_Discriminator_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
