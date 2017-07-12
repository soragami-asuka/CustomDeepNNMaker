//======================================
// バッチ正規化レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include"BatchNormalization_DATA.hpp"
#include"BatchNormalization_FUNC.hpp"
#include"BatchNormalization_Base.h"

#include"BatchNormalization_GPU.cuh"
#include"BatchNormalization_LayerData_GPU.cuh"


using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	BatchNormalization_GPU::BatchNormalization_GPU(Gravisbell::GUID guid, BatchNormalization_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct)
		:	BatchNormalization_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData				(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount		(0)				/**< 入力バッファ数 */
		,	outputBufferCount		(0)				/**< 出力バッファ数 */
		,	channeclBufferCount		(0)				/**< 1チャンネル当たりのバッファ数 */
		,	onLearnMode				(false)			/**< 学習処理中フラグ */
		,	learnCount				(0)				/**< 学習実行回数 */
		,	m_lppInputBuffer				(NULL)			/**< 演算時の入力データ */
		,	m_lppDOutputBufferPrev			(NULL)			/**< 入力誤差計算時の出力誤差データ */
	{
        cudnnCreate(&this->cudnnHandle);
		cudnnCreateTensorDescriptor(&this->paramTensorDesc);
        cudnnCreateTensorDescriptor(&this->inputTensorDesc);
        cudnnCreateTensorDescriptor(&this->outputTensorDesc);
	}
	/** デストラクタ */
	BatchNormalization_GPU::~BatchNormalization_GPU()
	{
        cudnnDestroyTensorDescriptor(this->inputTensorDesc);
        cudnnDestroyTensorDescriptor(this->outputTensorDesc);
		cudnnDestroyTensorDescriptor(this->paramTensorDesc);
        cudnnDestroy(this->cudnnHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 BatchNormalization_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode BatchNormalization_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	BatchNormalization_LayerData_Base& BatchNormalization_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const BatchNormalization_LayerData_Base& BatchNormalization_GPU::GetLayerData()const
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
	ErrorCode BatchNormalization_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 学習用の変数を作成
		this->onLearnMode = true;
		this->learnCount = 0;
		this->lpTmpMean.resize(this->GetInputDataStruct().ch, 0.0f);
		this->lpTmpVariance.resize(this->GetInputDataStruct().ch, 0.0f);

		// パラメータ変化量
		this->lpDBias.resize(this->layerData.lpBias.size());
		this->lpDScale.resize(this->layerData.lpScale.size());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalization_GPU::PreProcessCalculate()
	{
		cudnnStatus_t err_cudnn;

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
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->GetBatchSize() * this->outputBufferCount);


		// 次元数を調べる
		S32 dataDim = 1 + 1 + 0;	// バッチ + チャンネル + 次元0
		std::vector<S32> dimInput;			// 入力データ構造
		std::vector<S32> dimInputStride;	// 入力データの各次元ごとのデータ数
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		std::vector<S32> dimParam;
		std::vector<S32> dimParamStride;
		if(this->GetInputDataStruct().z > 1)
		{
			dataDim = 1 + 1 + 3;	// バッチ + チャンネル + 次元3

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().z;
			dimInput[3] = this->GetInputDataStruct().y;
			dimInput[4] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().z;
			dimOutput[3] = this->GetOutputDataStruct().y;
			dimOutput[4] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3] * dimOutput[4];
			dimOutputStride[2] = dimOutput[3] * dimOutput[4];
			dimOutputStride[3] = dimOutput[4];
			dimOutputStride[4] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->GetInputDataStruct().ch;
			dimParam[2] = 1;
			dimParam[3] = 1;
			dimParam[4] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2] * dimParam[3] * dimParam[4];
			dimParamStride[1] = dimParam[2] * dimParam[3] * dimParam[4];
			dimParamStride[2] = dimParam[3] * dimParam[4];
			dimParamStride[3] = dimParam[4];
			dimParamStride[4] = 1;
		}
		else if(this->GetInputDataStruct().y > 1 || this->GetInputDataStruct().x)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().y;
			dimInput[3] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;
			
			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().y;
			dimOutput[3] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3];
			dimOutputStride[2] = dimOutput[3];
			dimOutputStride[3] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->GetInputDataStruct().ch;
			dimParam[2] = 1;
			dimParam[3] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2] * dimParam[3];
			dimParamStride[1] = dimParam[2] * dimParam[3];
			dimParamStride[2] = dimParam[3];
			dimParamStride[3] = 1;
		}
		else if(this->GetInputDataStruct().x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize();
			dimInput[1] = this->GetInputDataStruct().ch;
			dimInput[2] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;
			
			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize();
			dimOutput[1] = this->GetOutputDataStruct().ch;
			dimOutput[2] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2];
			dimOutputStride[1] = dimOutput[2];
			dimOutputStride[2] = 1;

			dimParam.resize(dataDim);
			dimParam[0] = 1;
			dimParam[1] = this->GetInputDataStruct().ch;
			dimParam[2] = 1;

			dimParamStride.resize(dataDim);
			dimParamStride[0] = dimParam[1] * dimParam[2];
			dimParamStride[1] = dimParam[2];
			dimParamStride[2] = 1;
		}
		else
		{
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}


		// CUDNNの入力データ構造を作成
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->inputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimInputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNNの出力データ構造を作成
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNNのパラメータデータ構造を作成
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->paramTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimParam[0],
			&dimParamStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		return ErrorCode::ERROR_CODE_NONE;
	}

	
	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode BatchNormalization_GPU::PreProcessLoop()
	{
		switch(this->GetProcessType())
		{
		case ProcessType::PROCESSTYPE_LEARN:
			{
				// 学習回数を初期化
				this->learnCount = 0;

				// 演算用の平均.分散を初期化
				cudaMemset(thrust::raw_pointer_cast(&this->layerData.lpMean[0]),	 0, sizeof(F32)*this->GetInputDataStruct().ch);
				cudaMemset(thrust::raw_pointer_cast(&this->layerData.lpVariance[0]), 0, sizeof(F32)*this->GetInputDataStruct().ch);

				cudaMemset(thrust::raw_pointer_cast(&this->lpLearnMean[0]),		0, sizeof(F32)*this->GetInputDataStruct().ch);
				cudaMemset(thrust::raw_pointer_cast(&this->lpLearnVariance[0]),	0, sizeof(F32)*this->GetInputDataStruct().ch);
			}
			break;
		case ProcessType::PROCESSTYPE_CALCULATE:
			{
				// 平均,分散を一時バッファに移す
				this->lpTmpMean = this->layerData.lpMean;
				this->lpTmpVariance = this->layerData.lpVariance;

				this->lpLearnMean = this->layerData.lpMean;
				this->lpLearnVariance = this->layerData.lpVariance;
			}
			break;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode BatchNormalization_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 入力バッファのアドレスを格納
		this->m_lppInputBuffer = i_lpInputBuffer;

		// 学習中ならば平均、分散を求める
		if(this->onLearnMode)
		{
			// 学習中の場合
			F32 alpha = 1.0f;
			F32 beta = 0.0f;

			std::vector<F32> lpVarianceLast(this->GetInputDataStruct().ch);
			for(U32 i=0; i<lpVarianceLast.size(); i++)
				lpVarianceLast[i] = this->layerData.lpVariance[i];

			// 平均、分散を学習用に移す
			this->lpLearnMean     = this->layerData.lpMean;
			this->lpLearnVariance = this->layerData.lpVariance;

			err_cudnn = cudnnBatchNormalizationForwardTraining(
				this->cudnnHandle,
				cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				this->inputTensorDesc,
				this->m_lppInputBuffer,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				this->paramTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
				thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
				(1.0 / (this->learnCount+1)),
				thrust::raw_pointer_cast(&this->lpLearnMean[0]),
				thrust::raw_pointer_cast(&this->lpLearnVariance[0]),
				max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
				thrust::raw_pointer_cast(&this->lpTmpMean[0]),
				thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}
		else
		{
			// 学習中でない場合
			F32 alpha = 1.0f;
			F32 beta = 0.0f;

			err_cudnn = cudnnBatchNormalizationForwardInference(
				this->cudnnHandle,
				cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
				&alpha,
				&beta,
				this->inputTensorDesc,
				this->m_lppInputBuffer,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]),
				this->paramTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
				thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
				thrust::raw_pointer_cast(&this->layerData.lpMean[0]),
				thrust::raw_pointer_cast(&this->layerData.lpVariance[0]),
				max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode BatchNormalization_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseOutputBuffer = this->GetOutputBuffer();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32) * outputBufferCount * batchSize, cudaMemcpyDeviceToHost);

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
	ErrorCode BatchNormalization_GPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 出力誤差バッファのアドレスを格納
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// 入力誤差バッファのアドレスを格納
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;
		if(this->m_lpDInputBuffer_d == NULL)
		{
			// 入力誤差バッファが存在しない場合学習ができないため、代替バッファを確保
			if(this->m_lpTemporaryDInputBuffer_d.size() != this->inputBufferCount * this->GetBatchSize())
				this->m_lpTemporaryDInputBuffer_d.resize(this->inputBufferCount * this->GetBatchSize());

			this->m_lpDInputBuffer_d = thrust::raw_pointer_cast(&this->m_lpTemporaryDInputBuffer_d[0]);
		}


		F32 alphaData = 1.0f;
		F32 betaData  = 0.0f;

		F32 alphaParam = 0.0f;
		F32 betaParam  = 1.0f;

		err_cudnn = cudnnBatchNormalizationBackward(
			this->cudnnHandle,
			cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
			&alphaData,
			&betaData,
			&alphaParam,
			&betaParam,
			this->inputTensorDesc,
			this->m_lppInputBuffer,
			this->outputTensorDesc,
			this->m_lppDOutputBufferPrev,
			this->inputTensorDesc,
			this->m_lpDInputBuffer_d,
			this->paramTensorDesc,
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->layerData.lpBias[0]),
			max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
			thrust::raw_pointer_cast(&this->lpTmpMean[0]),
			thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], this->m_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode BatchNormalization_GPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 出力誤差バッファのアドレスを格納
		this->m_lppDOutputBufferPrev = i_lppDOutputBuffer;

		// 入力誤差バッファのアドレスを格納
		this->m_lpDInputBuffer_d = o_lppDInputBuffer;
		if(this->m_lpDInputBuffer_d == NULL)
		{
			// 入力誤差バッファが存在しない場合学習ができないため、代替バッファを確保
			if(this->m_lpTemporaryDInputBuffer_d.size() != this->inputBufferCount * this->GetBatchSize())
				this->m_lpTemporaryDInputBuffer_d.resize(this->inputBufferCount * this->GetBatchSize());

			this->m_lpDInputBuffer_d = thrust::raw_pointer_cast(&this->m_lpTemporaryDInputBuffer_d[0]);
		}


		F32 alphaData = 1.0f;
		F32 betaData  = 0.0f;

		F32 alphaParam = 1.0F;
		F32 betaParam  = 0.0F;

		err_cudnn = cudnnBatchNormalizationBackward(
			this->cudnnHandle,
			cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
			&alphaData,
			&betaData,
			&alphaParam,
			&betaParam,
			this->inputTensorDesc,
			this->m_lppInputBuffer,
			this->outputTensorDesc,
			this->m_lppDOutputBufferPrev,
			this->inputTensorDesc,
			this->m_lpDInputBuffer_d,
			this->paramTensorDesc,
			thrust::raw_pointer_cast(&this->layerData.lpScale[0]),
			thrust::raw_pointer_cast(&this->lpDScale[0]),
			thrust::raw_pointer_cast(&this->lpDBias[0]),
			max(CUDNN_BN_MIN_EPSILON, this->layerData.layerStructure.epsilon),
			thrust::raw_pointer_cast(&this->lpTmpMean[0]),
			thrust::raw_pointer_cast(&this->lpTmpVariance[0]));
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_CALCULATE;

		// 平均、分散を更新
		this->layerData.lpMean = this->lpLearnMean;
		this->layerData.lpVariance = this->lpLearnVariance;

		// パラメータを更新
		if(this->layerData.m_pOptimizer_scale)
			this->layerData.m_pOptimizer_scale->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpScale[0]), thrust::raw_pointer_cast(&this->lpDScale[0]));
		if(this->layerData.m_pOptimizer_bias)
			this->layerData.m_pOptimizer_bias->UpdateParameter(thrust::raw_pointer_cast(&this->layerData.lpBias[0]), thrust::raw_pointer_cast(&this->lpDBias[0]));

		// 学習処理の実行回数をカウントアップ
		this->learnCount++;


#ifdef _DEBUG
		std::vector<float> lpMean_h(this->layerData.lpMean.size());
		cudaMemcpy(&lpMean_h[0], thrust::raw_pointer_cast(&this->layerData.lpMean[0]), sizeof(float)*this->layerData.lpMean.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpVariance_h(this->layerData.lpVariance.size());
		cudaMemcpy(&lpVariance_h[0], thrust::raw_pointer_cast(&this->layerData.lpVariance[0]), sizeof(float)*this->layerData.lpVariance.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpDScale_h(this->lpDBias.size());
		cudaMemcpy(&lpDScale_h[0], thrust::raw_pointer_cast(&this->lpDScale[0]), sizeof(float)*lpDScale_h.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpDBias_h(this->lpDBias.size());
		cudaMemcpy(&lpDBias_h[0], thrust::raw_pointer_cast(&this->lpDBias[0]), sizeof(float)*lpDBias_h.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpScale_h(this->layerData.lpScale.size());
		cudaMemcpy(&lpScale_h[0], thrust::raw_pointer_cast(&this->layerData.lpScale[0]), sizeof(float)*lpScale_h.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpBias_h(this->layerData.lpBias.size());
		cudaMemcpy(&lpBias_h[0], thrust::raw_pointer_cast(&this->layerData.lpBias[0]), sizeof(float)*lpBias_h.size(), cudaMemcpyDeviceToHost);

#endif

#ifdef _DEBUG
		std::vector<float> lpTmpInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpInputBuffer[0], this->m_lppInputBuffer, sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpOutputBuffer[0], thrust::raw_pointer_cast(&this->lpOutputBuffer[0]), sizeof(float)*lpTmpInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpTmpDOutputBuffer[0], i_lppDOutputBuffer, sizeof(float)*lpTmpDOutputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<float> lpTmpDInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpTmpDInputBuffer[0], o_lppDInputBuffer, sizeof(float)*lpTmpDInputBuffer.size(), cudaMemcpyDeviceToHost);
#endif

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER BatchNormalization_GPU::GetDInputBuffer()const
	{
		return this->m_lpDInputBuffer_d;
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode BatchNormalization_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		CONST_BATCH_BUFFER_POINTER lppUseDInputBuffer = this->GetDInputBuffer();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32) * inputBufferCount * batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
