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

#include"Library/NeuralNetwork/Optimizer.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;

#define TEMPORARY_MEMORY_MAX	(100 * 1024 * 1024)

#define WORKSPACE_CODE			L"WorkSpace"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {


	/** コンストラクタ */
	Convolution_GPU::Convolution_GPU(Gravisbell::GUID guid, Convolution_LayerData_GPU& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	Convolution_Base	(guid, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
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
		,	temporaryMemoryManager	(i_temporaryMemoryManager)
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



	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessLearn()
	{
		ErrorCode errorCode = this->PreProcessCalculate();
		if(errorCode != ErrorCode::ERROR_CODE_NONE)
			return errorCode;


		// パラメータ変化量のバッファを確保
		this->lpDBias.resize(this->layerData.pWeightData->GetBiasSize());
		this->lpDNeuron.resize(this->layerData.pWeightData->GetWeigthSize());


		return ErrorCode::ERROR_CODE_NONE;
	}


	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessCalculate()
	{
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

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->GetOutputDataStruct().ch;
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
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.z;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[4] = this->layerData.layerStructure.FilterSize.x;

			convDim = 3;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.z;
			dimPadding[1] = this->layerData.layerStructure.Padding.y;
			dimPadding[2] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.Dilation.z;
			dimUpscale[1] = this->layerData.layerStructure.Dilation.y;
			dimUpscale[2] = this->layerData.layerStructure.Dilation.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.z;
			dimStride[1] = this->layerData.layerStructure.Stride.y;
			dimStride[2] = this->layerData.layerStructure.Stride.x;
		}
		else if(this->GetInputDataStruct().y > 1 || this->GetInputDataStruct().x >= 1)
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

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->GetOutputDataStruct().ch;
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
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.y;
			dimFilter[3] = this->layerData.layerStructure.FilterSize.x;

			convDim = 2;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.y;
			dimPadding[1] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.Dilation.y;
			dimUpscale[1] = this->layerData.layerStructure.Dilation.x;

			dimStride.resize(convDim);
			dimStride[0] = this->layerData.layerStructure.Stride.y;
			dimStride[1] = this->layerData.layerStructure.Stride.x;
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

			dimBias.resize(dataDim);
			dimBias[0] = 1;
			dimBias[1] = this->GetOutputDataStruct().ch;
			dimBias[2] = 1;

			dimBiasStride.resize(dataDim);
			dimBiasStride[0] = dimBias[1] * dimBias[2];
			dimBiasStride[1] = dimBias[2];
			dimBiasStride[2] = 1;

			dimOutput.resize(dataDim);

			filterDim = 1 + 1 + 1;	// 入力チャンネル + 出力チャンネル + 次元3

			dimFilter.resize(filterDim);
			dimFilter[0] = this->GetOutputDataStruct().ch;
			dimFilter[1] = this->GetInputDataStruct().ch;
			dimFilter[2] = this->layerData.layerStructure.FilterSize.x;

			convDim = 1;	// 次元3

			dimPadding.resize(convDim);
			dimPadding[0] = this->layerData.layerStructure.Padding.x;

			dimUpscale.resize(convDim);
			dimUpscale[0] = this->layerData.layerStructure.Dilation.x;

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
		if(outputBatchSize != this->GetBatchSize())
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
			cudnnConvolutionFwdPreference_t::CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,	// メモリの使用量無制限で最速のアルゴリズムを調べる
			TEMPORARY_MEMORY_MAX,										// 使用可能なメモリの上限
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
			cudnnConvolutionBwdDataPreference_t::CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,	// メモリの使用量無制限で最速のアルゴリズムを調べる
			TEMPORARY_MEMORY_MAX,																				// 使用可能なメモリの上限
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
			cudnnConvolutionBwdFilterPreference_t::CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,	// メモリの使用量無制限で最速のアルゴリズムを調べる
			TEMPORARY_MEMORY_MAX,																					// 使用可能なメモリの上限
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
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), WORKSPACE_CODE, (U32)max(workSpaceSizeByte_forward, max(workSpaceSizeByte_backwardData, workSpaceSizeByte_backwardFilter)));

		// バイアスのデータ構造を設定
		err_cudnn = cudnnSetTensorNdDescriptor(
			this->biasTensorDesc,
			CUDNN_DATA_FLOAT,
			dataDim,
			&dimBias[0],
			&dimBiasStride[0]);

		return ErrorCode::ERROR_CODE_NONE;
	}


	/** ループの初期化処理.データセットの実行開始前に実行する
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode Convolution_GPU::PreProcessLoop()
	{
		return Gravisbell::ErrorCode::ERROR_CODE_NONE;
	}



	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode Convolution_GPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		if(this->GetProcessType() == ProcessType::PROCESSTYPE_LEARN && this->GetRuntimeParameterByStructure().UpdateWeigthWithOutputVariance)
		{
			// ※とりあえずCPU側で処理.
			// 基本的に1回しか通らないから処理負荷に影響は与えない・・・はず
			// 超手抜き


			U32 PROCTIME_MAX = 5;			// 実行最大値
			F32	VARIANCE_TOLERANCE = 0.1f;	// 分散交差(許容範囲)

			std::vector<F32> lpTmpOutputBuffer(this->GetBatchSize() * this->outputBufferCount);

			// バッファを確保
			thrust::device_vector<F32> lpTmpWeight_d(this->layerData.pWeightData->GetWeigthSize());
			thrust::device_vector<F32> lpTmpBias_d(this->layerData.pWeightData->GetBiasSize());

			// バッファをコピー
			cudaMemcpy(thrust::raw_pointer_cast(&lpTmpWeight_d[0]), this->layerData.pWeightData->GetWeight(), sizeof(F32)*lpTmpWeight_d.size(), cudaMemcpyDeviceToDevice);
			cudaMemcpy(thrust::raw_pointer_cast(&lpTmpBias_d[0]),   this->layerData.pWeightData->GetBias(),   sizeof(F32)*lpTmpBias_d.size(), cudaMemcpyDeviceToDevice);


			U32 procTime = 0;
			do
			{
				// 演算を実行
				ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer, thrust::raw_pointer_cast(&lpTmpWeight_d[0]), thrust::raw_pointer_cast(&lpTmpBias_d[0]));
				if(err != ErrorCode::ERROR_CODE_NONE)
					return err;

				// バッファをコピー
				cudaMemcpy(&lpTmpOutputBuffer[0], o_lppOutputBuffer, sizeof(F32)*lpTmpOutputBuffer.size(), cudaMemcpyDeviceToHost);

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
					thrust::host_vector<F32> lpTmpNeuron = lpTmpWeight_d;
					thrust::host_vector<F32> lpTmpBias   = lpTmpBias_d;

					for(U32 neuronNum=0; neuronNum<lpTmpNeuron.size(); neuronNum++)
					{
						lpTmpNeuron[neuronNum] /= deviation;
					}
					for(U32 neuronNum=0; neuronNum<lpTmpBias.size(); neuronNum++)
					{
						lpTmpBias[neuronNum] /= deviation;
					}

					lpTmpWeight_d = lpTmpNeuron;
					lpTmpBias_d    = lpTmpBias;
				}

				procTime++;
			}while(procTime < 5);
			
			// 重みを更新
			this->layerData.pWeightData->SetData(thrust::raw_pointer_cast(&lpTmpWeight_d[0]), thrust::raw_pointer_cast(&lpTmpBias_d[0]));
		}
		else
		{
			ErrorCode err = this->Calculate_base(i_lppInputBuffer, o_lppOutputBuffer, this->layerData.pWeightData->GetWeight(), this->layerData.pWeightData->GetBias());
			if(err != ErrorCode::ERROR_CODE_NONE)
				return err;
		}


		return ErrorCode::ERROR_CODE_NONE;
	}
	ErrorCode Convolution_GPU::Calculate_base(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer, const F32* lpWeight, const F32* lpBias)
	{
		cudnnStatus_t err_cudnn;

		// 畳み込み処理
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;
			err_cudnn = cudnnConvolutionForward(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->filterDesc,
				lpWeight,
				this->convDesc,
				this->useForwardAlgorithm,
				this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE),
				this->temporaryMemoryManager.GetBufferSize(this->GetGUID(), WORKSPACE_CODE),
				&beta,
				this->outputTensorDesc,
				o_lppOutputBuffer);
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
				lpBias,
				&beta,
				this->outputTensorDesc,
				o_lppOutputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// 一時バッファ解放
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

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
	ErrorCode Convolution_GPU::CalculateDInput_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 入力誤差を計算
		if(o_lppDInputBuffer)
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardData(
				this->cudnnHandle,
				&alpha,
				this->filterDesc,
				this->layerData.pWeightData->GetWeight(),
				this->outputTensorDesc,
				i_lppDOutputBuffer,
				this->convDesc,
				this->useBackwardDataAlgorithm,
				this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE),
				this->temporaryMemoryManager.GetBufferSize(this->GetGUID(), WORKSPACE_CODE),
				&beta,
				this->inputTensorDesc,
				o_lppDInputBuffer);
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// 一時バッファ解放
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習処理を実行する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode Convolution_GPU::Training_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		cudnnStatus_t err_cudnn;

		// 入力誤差計算
		Gravisbell::ErrorCode errCode = this->CalculateDInput_device(i_lppInputBuffer, o_lppDInputBuffer, i_lppOutputBuffer, i_lppDOutputBuffer);
		if(errCode != ErrorCode::ERROR_CODE_NONE)
			return errCode;


		// フィルター変化量を計算
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardFilter(
				this->cudnnHandle,
				&alpha,
				this->inputTensorDesc,
				i_lppInputBuffer,
				this->outputTensorDesc,
				i_lppDOutputBuffer,
				this->convDesc,
				this->useBackwardFilterAlgorithm,
				this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), WORKSPACE_CODE),
				this->temporaryMemoryManager.GetBufferSize(this->GetGUID(), WORKSPACE_CODE),
				&beta,
				this->filterDesc,
				thrust::raw_pointer_cast(&this->lpDNeuron[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// バイアス変化量を計算
		{
			F32 alpha = 1.0f;
			F32 beta  = 0.0f;

			err_cudnn = cudnnConvolutionBackwardBias(
				this->cudnnHandle,
				&alpha,
				this->outputTensorDesc,
				i_lppDOutputBuffer,
				&beta,
				this->biasTensorDesc,
				thrust::raw_pointer_cast(&this->lpDBias[0]));
			if(err_cudnn != 0)
				return ErrorCode::ERROR_CODE_CUDA_CALCULATE;
		}

		// 変化量を反映
		this->layerData.pWeightData->UpdateData(thrust::raw_pointer_cast(&this->lpDNeuron[0]), thrust::raw_pointer_cast(&this->lpDBias[0]));

		// 一時バッファ解放
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), WORKSPACE_CODE);


		return ErrorCode::ERROR_CODE_NONE;
	}


} // Gravisbell;
} // Layer;
} // NeuralNetwork;
