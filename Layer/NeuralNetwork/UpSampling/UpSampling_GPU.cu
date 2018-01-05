//======================================
// 畳み込みニューラルネットワークの結合レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include"UpSampling_DATA.hpp"
#include"UpSampling_FUNC.hpp"
#include"UpSampling_Base.h"

#include"UpSampling_GPU.cuh"
#include"UpSampling_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	UpSampling_GPU::UpSampling_GPU(Gravisbell::GUID guid, UpSampling_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	UpSampling_Base		(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
		,	layerData			(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount	(0)		/**< 入力バッファ数 */
		,	outputBufferCount	(0)		/**< 出力バッファ数 */
		,	cudnnHandle			(NULL)
		,	inputTensorDesc		(NULL)
		,	outputTensorDesc	(NULL)
		,	filterDesc			(NULL)
		,	convDesc			(NULL)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateFilterDescriptor(&filterDesc);
		cudnnCreateConvolutionDescriptor(&convDesc);
	}
	/** デストラクタ */
	UpSampling_GPU::~UpSampling_GPU()
	{
		if(convDesc)			cudnnDestroyConvolutionDescriptor(convDesc);
		if(filterDesc)			cudnnDestroyFilterDescriptor(filterDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 UpSampling_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode UpSampling_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	UpSampling_LayerData_Base& UpSampling_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const UpSampling_LayerData_Base& UpSampling_GPU::GetLayerData()const
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
	ErrorCode UpSampling_GPU::PreProcessLearn()
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
	ErrorCode UpSampling_GPU::PreProcessCalculate()
	{
		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		cudnnStatus_t err_cudnn;

		// 次元数を調べる
		S32 dataDim = 1 + 1 + 0;	// バッチ + チャンネル + 次元0
		std::vector<S32> dimInput;			// 入力データ構造
		std::vector<S32> dimInputStride;	// 入力データの各次元ごとのデータ数
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		S32 filterDim = 0;			// フィルタ次元数	入力チャンネル + 出力チャンネル + 次元
		std::vector<S32> dimFilter;
		S32 convDim = 0;			// 畳み込み次元数	次元
		std::vector<S32> dimStride;
		std::vector<S32> dimDilation;
		std::vector<S32> dimPadding;
		if(this->GetInputDataStruct().z > 1)
		{
			dataDim = 1 + 1 + 3;

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

			filterDim = 1 + 1 + 2;	// 入力チャンネル + 出力チャンネル + 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.UpScale.y;
			dimFilter[3] = this->layerData.layerStructure.UpScale.x;

			convDim = 2;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = 0;
			dimPadding[1] = 0;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;
			dimDilation[1] = 1;

			dimStride.resize(convDim);
			dimStride[0] = 1;
			dimStride[1] = 1;

		}
		else if(this->GetInputDataStruct().y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->GetBatchSize() * this->GetInputDataStruct().ch;
			dimInput[1] = 1;
			dimInput[2] = this->GetInputDataStruct().y;
			dimInput[3] = this->GetInputDataStruct().x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;

			dimOutput.resize(dataDim);
			dimOutput[0] = this->GetBatchSize() * this->GetOutputDataStruct().ch;
			dimOutput[1] = 1;
			dimOutput[2] = this->GetOutputDataStruct().y;
			dimOutput[3] = this->GetOutputDataStruct().x;

			dimOutputStride.resize(dataDim);
			dimOutputStride[0] = dimOutput[1] * dimOutput[2] * dimOutput[3];
			dimOutputStride[1] = dimOutput[2] * dimOutput[3];
			dimOutputStride[2] = dimOutput[3];
			dimOutputStride[3] = 1;

			filterDim = 1 + 1 + 2;	// 入力チャンネル + 出力チャンネル + 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = 1;
			dimFilter[1] = 1;
			dimFilter[2] = this->layerData.layerStructure.UpScale.y;
			dimFilter[3] = this->layerData.layerStructure.UpScale.x;

			convDim = 2;	// 次元2

			dimPadding.resize(convDim);
			dimPadding[0] = 0;
			dimPadding[1] = 0;

			dimDilation.resize(convDim);
			dimDilation[0] = 1;
			dimDilation[1] = 1;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.UpScale.y;
			dimStride[1] = this->layerData.layerStructure.UpScale.x;
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

		}
		else
		{
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		}

		// CUDNNの入力データ構造を設定
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->inputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimInput[0],
			&dimInputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// CUDNNの出力データ構造を設定
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// フィルタサイズを設定
		err_cudnn = cudnnSetFilterNdDescriptor(
			this->filterDesc,
			CUDNN_DATA_FLOAT,
			CUDNN_TENSOR_NCHW,
			filterDim,
			&dimFilter[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 畳み込み処理設定
		err_cudnn = cudnnSetConvolutionNdDescriptor(
			this->convDesc,
			convDim,
			&dimPadding[0],
			&dimStride[0],
			&dimDilation[0],
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 最速のアルゴリズムを検索する(前方伝播)
		err_cudnn = cudnnGetConvolutionForwardAlgorithm(
			this->cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,	// メモリの使用量無制限で最速のアルゴリズムを調べる
			0,										// 使用可能なメモリの上限
			&this->useForwardAlgorithm
			);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 必要なメモリ量を調べる(前方伝播)
		size_t workSpaceSizeByte_forward;
		err_cudnn = cudnnGetConvolutionForwardWorkspaceSize(
			this->cudnnHandle,
			this->outputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->useForwardAlgorithm,
			&workSpaceSizeByte_forward);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// 最速のアルゴリズムを検索する(後方伝播-データ)
		err_cudnn = cudnnGetConvolutionBackwardDataAlgorithm(
			this->cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			cudnnConvolutionBwdDataPreference_t::CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,	// メモリの使用量無制限で最速のアルゴリズムを調べる
			0,																				// 使用可能なメモリの上限
			&this->useBackwardDataAlgorithm);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 必要なメモリ量を調べる(後方伝播-データ)
		size_t workSpaceSizeByte_backwardData;
		err_cudnn = cudnnGetConvolutionBackwardDataWorkspaceSize(
			this->cudnnHandle,
			this->filterDesc,
			this->inputTensorDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->useBackwardDataAlgorithm,
			&workSpaceSizeByte_backwardData);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// 処理用バッファの確保
		this->workSpace.resize(max(workSpaceSizeByte_forward, workSpaceSizeByte_backwardData));


		// フィルタバッファを作成して初期化
		filter.resize(
			this->layerData.layerStructure.UpScale.x * this->layerData.layerStructure.UpScale.y * this->layerData.layerStructure.UpScale.z,
			0.0f);
		for(U32 z=0; z<this->layerData.layerStructure.UpScale.z; z++)
		{
			U32 zOffset = z * this->layerData.layerStructure.UpScale.y * this->layerData.layerStructure.UpScale.x;

			for(U32 y=0; y<this->layerData.layerStructure.UpScale.y; y++)
			{
				U32 yOffset = y * this->layerData.layerStructure.UpScale.x;

				for(U32 x=0; x<this->layerData.layerStructure.UpScale.x; x++)
				{
					U32 offset = zOffset + yOffset + x;

					switch(this->layerData.layerStructure.PaddingType)
					{
					case UpSampling::LayerStructure::PaddingType_value:
						{
							filter[offset] = 1.0f;
						}
						break;
					case UpSampling::LayerStructure::PaddingType_zero:
						{
							if(z==0 && y==0 && x==0)
								filter[offset] = 1.0f;
							else
								filter[offset] = 0.0f;
						}
						break;
					}
				}
			}
		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode UpSampling_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode UpSampling_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 出力バッファをクリア
		cudaMemset(
			o_lppOutputBuffer,
			0,
			this->outputBufferCount * this->GetBatchSize() * sizeof(F32));

		// 入力バッファを出力にコピー
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardData(
				this->cudnnHandle,
				&alpha,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->filter[0]),
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->convDesc,
				this->useBackwardDataAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->outputTensorDesc,
				o_lppOutputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

#ifdef _DEBUG
		std::vector<F32> lpDebugInputBuffer(this->GetBatchSize() * this->inputBufferCount);
		cudaMemcpy(&lpDebugInputBuffer[0], i_lppInputBuffer, sizeof(F32)*lpDebugInputBuffer.size(), cudaMemcpyDeviceToHost);

		std::vector<F32> lpDebugOutputBuffer(this->GetBatchSize() * this->outputBufferCount);
		cudaMemcpy(&lpDebugOutputBuffer[0], o_lppOutputBuffer, sizeof(F32)*lpDebugOutputBuffer.size(), cudaMemcpyDeviceToHost);
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
	ErrorCode UpSampling_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 入力誤差計算
		if(o_lppDInputBuffer)
		{
			// 入力誤差バッファのクリア
			cudaMemset(
				o_lppDInputBuffer,
				0,
				sizeof(F32)*this->inputBufferCount*this->GetBatchSize());

			{
				F32 alpha = 1.0f;
				F32 beta  = 0.0f;
				err_cudnn = cudnnConvolutionForward(
					this->cudnnHandle,
					&alpha,
					this->outputTensorDesc,
					i_lppDOutputBuffer,
					this->filterDesc,
					thrust::raw_pointer_cast(&this->filter[0]),
					this->convDesc,
					this->useForwardAlgorithm,
					thrust::raw_pointer_cast(&this->workSpace[0]),
					this->workSpace.size(),
					&beta,
					this->inputTensorDesc,
					o_lppDInputBuffer);
				if(err_cudnn != 0)
					return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
			}

		}

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode UpSampling_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
