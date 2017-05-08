//======================================
// 畳み込みニューラルネットワークの結合レイヤー
// GPU処理用
//======================================
#include"stdafx.h"

#include"Convolution_DATA.hpp"
#include"Convolution_FUNC.hpp"
#include"Convolution_Base.h"

#include"Convolution_GPU.cuh"
#include"Convolution_LayerData_GPU.cuh"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Convolution_GPU::Convolution_GPU(Gravisbell::GUID guid, Convolution_LayerData_GPU& i_layerData)
		:	Convolution_Base	(guid)
		,	layerData			(i_layerData)	/**< レイヤーデータ */
		,	inputBufferCount	(0)		/**< 入力バッファ数 */
		,	neuronCount			(0)		/**< ニューロン数 */
		,	outputBufferCount	(0)		/**< 出力バッファ数 */
		,	cudnnHandle			(NULL)
		,	inputTensorDesc		(NULL)
		,	outputTensorDesc	(NULL)
		,	biasTensorDesc		(NULL)
		,	filterDesc			(NULL)
		,	convDesc			(NULL)
	{
		cudnnCreate(&cudnnHandle);
		cudnnCreateTensorDescriptor(&inputTensorDesc);
		cudnnCreateTensorDescriptor(&outputTensorDesc);
		cudnnCreateTensorDescriptor(&biasTensorDesc);
		cudnnCreateFilterDescriptor(&filterDesc);
		cudnnCreateConvolutionDescriptor(&convDesc);
	}
	/** デストラクタ */
	Convolution_GPU::~Convolution_GPU()
	{
		if(convDesc)			cudnnDestroyConvolutionDescriptor(convDesc);
		if(filterDesc)			cudnnDestroyFilterDescriptor(filterDesc);
		if(biasTensorDesc)		cudnnDestroyTensorDescriptor(biasTensorDesc);
		if(outputTensorDesc)	cudnnDestroyTensorDescriptor(outputTensorDesc);
		if(inputTensorDesc)		cudnnDestroyTensorDescriptor(inputTensorDesc);
		if(cudnnHandle)			cudnnDestroy(cudnnHandle);
	}


	//================================
	// 基本処理
	//================================
	/** レイヤー種別の取得 */
	U32 Convolution_GPU::GetLayerKind()const
	{
		return Layer::ELayerKind::LAYER_KIND_GPU | GetLayerKindBase();
	}

	/** 初期化. 各ニューロンの値をランダムに初期化
		@return	成功した場合0 */
	ErrorCode Convolution_GPU::Initialize(void)
	{
		return this->layerData.Initialize();
	}


	//===========================
	// レイヤーデータ関連
	//===========================
	/** レイヤーデータを取得する */
	Convolution_LayerData_Base& Convolution_GPU::GetLayerData()
	{
		return this->layerData;
	}
	const Convolution_LayerData_Base& Convolution_GPU::GetLayerData()const
	{
		return this->layerData;
	}


	//===========================
	// レイヤー保存
	//===========================
	/** レイヤーをバッファに書き込む.
		@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
		@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
	S32 Convolution_GPU::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		return this->layerData.WriteToBuffer(o_lpBuffer);
	}


	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessLearn(unsigned int batchSize)
	{
		ErrorCode errorCode = this->PreProcessCalculate(batchSize);
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;

		// 入力差分バッファを作成
		this->lpDInputBuffer.resize(this->batchSize * this->inputBufferCount);

		// ニューロン/バイアスの誤差を一時保存するバッファを作成
		this->lpDBias.resize(this->layerData.lpBias_d.size());
		this->lppDNeuron.resize(this->layerData.lppNeuron_d.size());

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessCalculate(unsigned int batchSize)
	{
		this->batchSize = batchSize;

		// 入力バッファ数を確認
		this->inputBufferCount = this->GetInputBufferCount();
		if(this->inputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_INPUT_COUNT;
		
		// ニューロン数を確認
		this->neuronCount = this->layerData.layerStructure.Output_Channel;
		if(this->neuronCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_NEURON_COUNT;

		// 出力バッファ数を確認
		this->outputBufferCount = this->GetOutputBufferCount();
		if(this->outputBufferCount == 0)
			return ErrorCode::ERROR_CODE_FRAUD_OUTPUT_COUNT;

		cudnnStatus_t err_cudnn;

		// 次元数を調べる
		S32 dataDim = 1 + 1 + 0;	// バッチ + チャンネル + 次元0
		std::vector<S32> dimInput;			// 入力データ構造
		std::vector<S32> dimInputStride;	// 入力データの各次元ごとのデータ数
		std::vector<S32> dimBias;
		std::vector<S32> dimBiasStride;
		std::vector<S32> dimOutput;
		std::vector<S32> dimOutputStride;
		S32 filterDim = 0;			// フィルタ次元数	入力チャンネル + 出力チャンネル + 次元
		std::vector<S32> dimFilter;
		S32 convDim = 0;			// 畳み込み次元数	次元
		std::vector<S32> dimStride;
		std::vector<S32> dimUpscale;
		std::vector<S32> dimPadding;
		if(this->layerData.inputDataStruct.z > 1)
		{
			dataDim = 1 + 1 + 3;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.z;
			dimInput[2] = this->layerData.inputDataStruct.y;
			dimInput[3] = this->layerData.inputDataStruct.x;
			dimInput[4] = this->layerData.inputDataStruct.ch;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[1] = dimInput[2] * dimInput[3] * dimInput[4];
			dimInputStride[2] = dimInput[3] * dimInput[4];
			dimInputStride[3] = dimInput[4];
			dimInputStride[4] = 1;

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->layerData.GetOutputDataStruct().ch;
			dimBias[2] = 1;
			dimBias[3] = 1;
			dimBias[4] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2] * dimBias[3] * dimBias[4];
			dimBiasStride[1] = dimBias[2] * dimBias[3] * dimBias[4];
			dimBiasStride[2] = dimBias[3] * dimBias[4];
			dimBiasStride[3] = dimBias[4];
			dimBiasStride[4] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 3;	// 入力チャンネル + 出力チャンネル + 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.GetOutputDataStruct().ch;
			dimFilter[1] = this->layerData.inputDataStruct.ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.z;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[4] = this->layerData.layerStructure.FilterSize.x;

			convDim = 3;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.z;
			dimPadding[1] = this->layerData.layerStructure.Padding.y;
			dimPadding[2] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.UpScale.z;
			dimUpscale[1] = this->layerData.layerStructure.UpScale.y;
			dimUpscale[2] = this->layerData.layerStructure.UpScale.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.z;
			dimStride[1] = this->layerData.layerStructure.Stride.y;
			dimStride[2] = this->layerData.layerStructure.Stride.x;
		}
		else if(this->layerData.inputDataStruct.y > 1)
		{
			dataDim = 1 + 1 + 2;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.y;
			dimInput[3] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2] * dimInput[3];
			dimInputStride[1] = dimInput[2] * dimInput[3];
			dimInputStride[2] = dimInput[3];
			dimInputStride[3] = 1;

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->layerData.GetOutputDataStruct().ch;
			dimBias[2] = 1;
			dimBias[3] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2] * dimBias[3];
			dimBiasStride[1] = dimBias[2] * dimBias[3];
			dimBiasStride[2] = dimBias[3];
			dimBiasStride[3] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 2;	// 入力チャンネル + 出力チャンネル + 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.GetOutputDataStruct().ch;
			dimFilter[1] = this->layerData.inputDataStruct.ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.x;

			convDim = 2;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.y;
			dimPadding[1] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.UpScale.y;
			dimUpscale[1] = this->layerData.layerStructure.UpScale.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.y;
			dimStride[1] = this->layerData.layerStructure.Stride.x;
		}
		else if(this->layerData.inputDataStruct.x > 1)
		{
			dataDim = 1 + 1 + 1;

			dimInput.resize(dataDim);
			dimInput[0] = this->batchSize;
			dimInput[1] = this->layerData.inputDataStruct.ch;
			dimInput[2] = this->layerData.inputDataStruct.x;

			dimInputStride.resize(dataDim);
			dimInputStride[0] = dimInput[1] * dimInput[2];
			dimInputStride[1] = dimInput[2];
			dimInputStride[2] = 1;

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->layerData.GetOutputDataStruct().ch;
			dimBias[2] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2];
			dimBiasStride[1] = dimBias[2];
			dimBiasStride[2] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 1;	// 入力チャンネル + 出力チャンネル + 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->layerData.GetOutputDataStruct().ch;
			dimFilter[1] = this->layerData.inputDataStruct.ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.x;

			convDim = 1;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.UpScale.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.x;
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
			&dimUpscale[0],
			CUDNN_CROSS_CORRELATION,
			CUDNN_DATA_FLOAT);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 出力データ構造を取得
        err_cudnn = cudnnGetConvolutionNdForwardOutputDim(
			this->convDesc,
			this->inputTensorDesc,
			this->filterDesc,
			dataDim,
			&dimOutput[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNNの出力データ構造とGravisbellの出力データ構造が一致することを確認
		Gravisbell::Vector3D<S32> outputVector;
		S32 outputBatchSize = dimOutput[0];
		S32 outputCh = dimOutput[1];
		if(dataDim == 5)
		{
			outputVector.z = dimOutput[2];
			outputVector.y = dimOutput[3];
			outputVector.x = dimOutput[4];
		}
		else if(dataDim == 4)
		{
			outputVector.z = 1;
			outputVector.y = dimOutput[2];
			outputVector.x = dimOutput[3];
		}
		else if(dataDim == 3)
		{
			outputVector.z = 1;
			outputVector.y = 1;
			outputVector.x = dimOutput[2];
		}
		if(outputBatchSize != this->batchSize)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputCh != this->GetOutputDataStruct().ch)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.z != this->GetOutputDataStruct().z)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.y != this->GetOutputDataStruct().y)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;
		if(outputVector.x != this->GetOutputDataStruct().x)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// CUDNNの出力データ構造を設定
		dimOutputStride.resize(dataDim);
		for(S32 i=0; i<dataDim; i++)
		{
			dimOutputStride[i] = 1;
			for(S32 j=i+1; j<dataDim; j++)
				dimOutputStride[i] *= dimOutput[j];
		}
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->outputTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimOutput[0],
			&dimOutputStride[0]);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_ALLOCATION_MEMORY;

		// 最速のアルゴリズムを検索する(前方伝播)
		err_cudnn = cudnnGetConvolutionForwardAlgorithm(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
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
			this->inputTensorDesc,
			this->filterDesc,
			this->convDesc,
			this->outputTensorDesc,
			this->useForwardAlgorithm,
			&workSpaceSizeByte_forward);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// 最速のアルゴリズムを検索する(後方伝播-データ)
		err_cudnn = cudnnGetConvolutionBackwardDataAlgorithm(
			this->cudnnHandle,
			this->filterDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
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
			this->outputTensorDesc,
			this->convDesc,
			this->inputTensorDesc,
			this->useBackwardDataAlgorithm,
			&workSpaceSizeByte_backwardData);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// 最速のアルゴリズムを検索する(後方伝播-データ)
		err_cudnn = cudnnGetConvolutionBackwardFilterAlgorithm(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			cudnnConvolutionBwdFilterPreference_t::CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,	// メモリの使用量無制限で最速のアルゴリズムを調べる
			0,																					// 使用可能なメモリの上限
			&this->useBackwardFilterAlgorithm);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;

		// 必要なメモリ量を調べる(後方伝播-データ)
		size_t workSpaceSizeByte_backwardFilter;
		err_cudnn = cudnnGetConvolutionBackwardFilterWorkspaceSize(
			this->cudnnHandle,
			this->inputTensorDesc,
			this->outputTensorDesc,
			this->convDesc,
			this->filterDesc,
			this->useBackwardFilterAlgorithm,
			&workSpaceSizeByte_backwardFilter);
		if(err_cudnn != 0)
			return ErrorCode::ERROR_CODE_CUDA_INITIALIZE;


		// 処理用バッファの確保
		this->workSpace.resize(max(workSpaceSizeByte_forward, max(workSpaceSizeByte_backwardData, workSpaceSizeByte_backwardFilter)));

		// 出力バッファを作成
		this->lpOutputBuffer.resize(this->batchSize * this->outputBufferCount);

		// バイアスのデータ構造を設定
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->biasTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimBias[0],
			&dimBiasStride[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習ループの初期化処理.データセットの学習開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessLearnLoop(const SettingData::Standard::IData& data)
	{
		if(this->pLearnData != NULL)
			delete this->pLearnData;
		this->pLearnData = data.Clone();

		// 学習係数
		{
			auto pItem = dynamic_cast<const Gravisbell::SettingData::Standard::IItem_Float*>(data.GetItemByID(L"LearnCoeff"));
			if(pItem)
				this->learnData.LearnCoeff = pItem->GetValue();
			else
				this->learnData.LearnCoeff = 1.0f;
		}

		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}
	/** 演算ループの初期化処理.データセットの演算開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessCalculateLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Convolution_GPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 入力バッファを保存
		this->m_lppInputBuffer_d = i_lpInputBuffer;

		// 畳み込み処理
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;
			err_cudnn = cudnnConvolutionForward(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				i_lpInputBuffer,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),
				this->convDesc,
				this->useForwardAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// バイアスを追加
		{
			F32 alpha = 1.0f;
			F32 beta  = 1.0f;

			err_cudnn = cudnnAddTensor(
				this->cudnnHandle,
				&alpha,
				this->biasTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]),
				&beta,
				this->outputTensorDesc,
				thrust::raw_pointer_cast(&this->lpOutputBuffer[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Convolution_GPU::GetOutputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpOutputBuffer[0]);
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode Convolution_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 学習処理
	//================================
	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Convolution_GPU::CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lpDOutputBufferPrev)
	{
		cudnnStatus_t err_cudnn;

		// 出力誤差バッファのアドレスを格納
		this->m_lppDOutputBuffer_d = i_lpDOutputBufferPrev;

		// 入力誤差を計算
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardData(
				this->cudnnHandle,
				&alpha,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]),
				this->outputTensorDesc,
				this->m_lppDOutputBuffer_d,
				this->convDesc,
				this->useBackwardDataAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->inputTensorDesc,
				thrust::raw_pointer_cast(&this->lpDInputBuffer[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// フィルター変化量を計算
		{
			F32 alpha = this->learnData.LearnCoeff;
			F32 beta  = 1.0f;

			err_cudnn = cudnnConvolutionBackwardFilter(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				this->m_lppInputBuffer_d,
				this->outputTensorDesc,
				this->m_lppDOutputBuffer_d,
				this->convDesc,
				this->useBackwardFilterAlgorithm,
				thrust::raw_pointer_cast(&this->workSpace[0]),
				this->workSpace.size(),
				&beta,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->layerData.lppNeuron_d[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// バイアス変化量を計算
		{
			F32 alpha = this->learnData.LearnCoeff;
			F32 beta  = 1.0f;

			err_cudnn = cudnnConvolutionBackwardBias(
				this->cudnnHandle,
				&alpha,
				this->outputTensorDesc,
				this->m_lppDOutputBuffer_d,
				&beta,
				this->biasTensorDesc,
				thrust::raw_pointer_cast(&this->layerData.lpBias_d[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 学習差分をレイヤーに反映させる.
		入力信号、出力信号は直前のCalculateの値を参照する.
		出力誤差差分、入力誤差差分は直前のCalculateLearnErrorの値を参照する. */
	ErrorCode Convolution_GPU::ReflectionLearnError(void)
	{

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分を取得する.
		配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
		@return	誤差差分配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER Convolution_GPU::GetDInputBuffer()const
	{
		return thrust::raw_pointer_cast(&this->lpDInputBuffer[0]);
	}
	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode Convolution_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, this->GetDInputBuffer(), sizeof(F32)*inputBufferCount*this->batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
