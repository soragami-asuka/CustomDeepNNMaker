//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// GPU処理
//======================================
#include"stdafx.h"

#include<algorithm>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU_base.cuh"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Library/Common/TemporaryMemoryManager.h"

// CUDA用
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda_runtime_api.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** コンストラクタ */
	FeedforwardNeuralNetwork_GPU_base::FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1), Common::CreateTemporaryMemoryManagerGPU())
	{
	}
	/** コンストラクタ */
	FeedforwardNeuralNetwork_GPU_base::FeedforwardNeuralNetwork_GPU_base(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1), i_temporaryMemoryManager)
	{
	}
	/** デストラクタ */
	FeedforwardNeuralNetwork_GPU_base::~FeedforwardNeuralNetwork_GPU_base()
	{
	}


	//====================================
	// 入力誤差バッファ関連
	//====================================
	/** 入力誤差バッファの総数を設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::SetDInputBufferCount(U32 i_DInputBufferCount)
	{
		this->lpDInputBuffer.resize(i_DInputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 入力誤差バッファのサイズを設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpDInputBuffer[i_DInputBufferNo].resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 入力誤差バッファを取得する */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_base::GetDInputBuffer_d(U32 i_DInputBufferNo)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_DInputBufferNo][0]);
	}
	/** 入力誤差バッファを取得する */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_base::GetDInputBuffer_d(U32 i_DInputBufferNo)const
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_DInputBufferNo][0]);
	}


	//====================================
	// 入出力バッファ関連
	//====================================
	/** 出力データバッファを取得する.
		配列の要素数はGetOutputBufferCountの戻り値.
		@return 出力データ配列の先頭ポインタ */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_base::GetOutputBuffer()const
	{
		return &this->lpOutputBuffer_h[0];
	}
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, this->GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}


	//================================
	// 演算処理
	//================================
	/** 演算前処理を実行する.(学習用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はPreProcessLearnLoop以降の処理は実行不可. */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::PreProcessLearn(U32 batchSize)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::PreProcessLearn(batchSize);

		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"input[0]", sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize());
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize());

		// 出力バッファの確保
		this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

		return err;
	}
	/** 演算前処理を実行する.(演算用)
		@param batchSize	同時に演算を行うバッチのサイズ.
		NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
		失敗した場合はCalculate以降の処理は実行不可. */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::PreProcessCalculate(unsigned int batchSize)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::PreProcessCalculate(batchSize);

		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"input[0]", sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize());
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"dinput[0]", sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize());
		this->temporaryMemoryManager.SetBufferSize(this->GetGUID(), L"doutput[0]", sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize());

		// 出力バッファの確保
		this->lpOutputBuffer_h.resize(this->GetOutputBufferCount() * this->GetBatchSize());

		return err;
	}


	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

		// 出力バッファをコピー
		if(err == ErrorCode::ERROR_CODE_NONE)
		{
			if(o_lppOutputBuffer)
			{
				cudaMemcpy(o_lppOutputBuffer, this->outputLayer.GetOutputBuffer_d(), sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToDevice);
			}
			cudaMemcpy(&this->lpOutputBuffer_h[0], this->outputLayer.GetOutputBuffer_d(), sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
		}

		return err;
	}
	/** 演算処理を実行する.
		ホストメモリが渡される
		@param i_lppInputBuffer		入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
		@param o_lppOutputBuffer	出力データバッファ. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		// 入力バッファをコピー
		if(this->lpInputBuffer.empty())
			this->lpInputBuffer.resize(this->GetInputBufferCount() * this->GetBatchSize());
		memcpy(&this->lpInputBuffer[0], i_lppInputBuffer, sizeof(F32)*this->lpInputBuffer.size());

		// 入力バッファをデバイスにコピー
		F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
		cudaMemcpy(lppInputBuffer, &this->lpInputBuffer[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

		// 演算
		Gravisbell::ErrorCode err = this->Calculate_device(lppInputBuffer, NULL);

		if(o_lppOutputBuffer)
			this->GetOutputBuffer(o_lppOutputBuffer);

		// バッファを開放
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");

		return err;
	}
	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		return this->Calculate(i_lpInputBuffer, NULL);
	}

	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::CalculateDInput(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力バッファをデバイスにコピー
		F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
		cudaMemcpy(lppInputBuffer, &this->lpInputBuffer[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

		// 出力誤差バッファをデバイスにコピー
		F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
		cudaMemcpy(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

		// 入力誤差バッファを確保
		F32* lppDInputBuffer = NULL;
		if(o_lppDInputBuffer)
			lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

		// 演算
		Gravisbell::ErrorCode err = CalculateDInput_device(lppInputBuffer, lppDInputBuffer, NULL, lppDOutputBuffer);

		// 入力誤差バッファをデバイスにコピー
		if(o_lppDInputBuffer)
		{
			cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
		}

		// バッファを開放
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

		return err;
	}
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->CalculateDInput(NULL, o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}



	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Training(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppOutputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		// 入力バッファをデバイスにコピー
		F32* lppInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"input[0]");
		cudaMemcpy(lppInputBuffer, &this->lpInputBuffer[0], sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

		// 出力誤差バッファをデバイスにコピー
		F32* lppDOutputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"doutput[0]");
		cudaMemcpy(lppDOutputBuffer, i_lppDOutputBuffer, sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize(), cudaMemcpyHostToDevice);

		// 入力誤差バッファを確保
		F32* lppDInputBuffer = NULL;
		if(o_lppDInputBuffer)
			lppDInputBuffer = (F32*)this->temporaryMemoryManager.ReserveBuffer(this->GetGUID(), L"dinput[0]");

		// 演算
		Gravisbell::ErrorCode err = Training_device(lppInputBuffer, o_lppDInputBuffer, NULL, lppDOutputBuffer);

		// 入力誤差バッファをデバイスにコピー
		if(o_lppDInputBuffer)
		{
			cudaMemcpy(o_lppDInputBuffer, lppDInputBuffer, sizeof(F32)*this->GetInputBufferCount()*this->GetBatchSize(), cudaMemcpyDeviceToHost);
			this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"dinput[0]");
		}

		// バッファを開放
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"input[0]");
		this->temporaryMemoryManager.RestoreBuffer(this->GetGUID(), L"doutput[0]");

		return err;
	}
	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_GPU_base::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return this->Training(NULL, o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

