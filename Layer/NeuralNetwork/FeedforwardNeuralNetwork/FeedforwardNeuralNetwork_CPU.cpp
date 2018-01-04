//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// CPU処理
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_CPU.h"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"

#include"Library/Common/TemporaryMemoryManager.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** コンストラクタ */
	FeedforwardNeuralNetwork_CPU::FeedforwardNeuralNetwork_CPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1), Common::CreateTemporaryMemoryManagerCPU())
	{
	}
	/** コンストラクタ */
	FeedforwardNeuralNetwork_CPU::FeedforwardNeuralNetwork_CPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1), i_temporaryMemoryManager)
	{
	}

	/** デストラクタ */
	FeedforwardNeuralNetwork_CPU::~FeedforwardNeuralNetwork_CPU()
	{
	}

	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 FeedforwardNeuralNetwork_CPU::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_CPU | Gravisbell::Layer::LAYER_KIND_HOSTMEMORY;
	}


	//====================================
	// 入力誤差バッファ関連
	//====================================
	/** 入力誤差バッファの総数を設定する */
	ErrorCode FeedforwardNeuralNetwork_CPU::SetDInputBufferCount(U32 i_DInputBufferCount)
	{
		this->lpDInputBuffer.resize(i_DInputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 入力誤差バッファのサイズを設定する */
	ErrorCode FeedforwardNeuralNetwork_CPU::ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpDInputBuffer[i_DInputBufferNo].resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 入力誤差バッファを取得する */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_CPU::GetDInputBuffer_d(U32 i_DInputBufferNo)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return &this->lpDInputBuffer[i_DInputBufferNo][0];
	}


	//====================================
	// 出力バッファ関連
	//====================================
	/** 出力バッファの総数を設定する */
	ErrorCode FeedforwardNeuralNetwork_CPU::SetOutputBufferCount(U32 i_outputBufferCount)
	{
		this->lpOutputBuffer.resize(i_outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力バッファのサイズを設定する */
	ErrorCode FeedforwardNeuralNetwork_CPU::ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpOutputBuffer[i_outputBufferNo].lpBuffer.resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力バッファの現在の使用者を取得する */
	GUID FeedforwardNeuralNetwork_CPU::GetReservedOutputBufferID(U32 i_outputBufferNo)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return GUID();

		return this->lpOutputBuffer[i_outputBufferNo].reserveLayerID;
	}
	/** 出力バッファを使用中にして取得する(処理デバイス依存) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_CPU::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return NULL;

		this->lpOutputBuffer[i_outputBufferNo].reserveLayerID = i_guid;

		return &this->lpOutputBuffer[i_outputBufferNo].lpBuffer[0];
	}
	/** 出力バッファを使用中にして取得する(処理デバイス依存)
		@param	i_outputBufferNo	出力バッファ番号
		@param	i_lppBuffer			バッファの初期化に使用するホストバッファ
		@param	i_bufferSize		初期化バッファのサイズ. */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_CPU::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid, CONST_BATCH_BUFFER_POINTER i_lppBuffer, U32 i_bufferSize)
	{
		if(i_outputBufferNo >= this->lpOutputBuffer.size())
			return NULL;

		if(this->lpOutputBuffer[i_outputBufferNo].reserveLayerID != i_guid)
		{
			memcpy(
				&this->lpOutputBuffer[i_outputBufferNo].lpBuffer[0],
				i_lppBuffer,
				sizeof(F32) * min(this->lpOutputBuffer[i_outputBufferNo].lpBuffer.size(), i_bufferSize));

			this->lpOutputBuffer[i_outputBufferNo].reserveLayerID = i_guid;
		}

		return &this->lpOutputBuffer[i_outputBufferNo].lpBuffer[0];
	}


	//====================================
	// 入出力バッファ関連
	//====================================
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_CPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		memcpy(o_lpOutputBuffer, FeedforwardNeuralNetwork_Base::GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	//================================
	// 演算処理
	//================================
	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FeedforwardNeuralNetwork_CPU::Calculate_device(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer, BATCH_BUFFER_POINTER o_lppOutputBuffer)
	{
		ErrorCode err = FeedforwardNeuralNetwork_Base::Calculate_device(i_lppInputBuffer, o_lppOutputBuffer);

		// 出力バッファをコピー
		if(err == ErrorCode::ERROR_CODE_NONE)
		{
			if(o_lppOutputBuffer)
			{
				memcpy(o_lppOutputBuffer, this->outputLayer.GetOutputBuffer_d(), sizeof(F32)*this->GetOutputBufferCount()*this->GetBatchSize());
			}
		}

		return err;
	}
	/** 演算処理を実行する.
		@param lpInputBuffer	入力データバッファ. GetInputBufferCountで取得した値の要素数が必要
		@return 成功した場合0が返る */
	ErrorCode FeedforwardNeuralNetwork_CPU::Calculate(CONST_BATCH_BUFFER_POINTER i_lpInputBuffer)
	{
		// 入力バッファをコピー
		if(this->lpInputBuffer.empty())
			this->lpInputBuffer.resize(this->GetInputBufferCount() * this->GetBatchSize());
		memcpy(&this->lpInputBuffer[0], i_lpInputBuffer, sizeof(F32)*this->lpInputBuffer.size());

		return this->Calculate_device(&this->lpInputBuffer[0], NULL);
	}

	//================================
	// 学習処理
	//================================
	/** 入力誤差計算をを実行する.学習せずに入力誤差を取得したい場合に使用する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	o_lppDInputBuffer	入力誤差差分格納先レイヤー.	[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_CPU::CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return  CalculateDInput_device(&this->lpInputBuffer[0], o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}
	/** 学習誤差を計算する.
		入力信号、出力信号は直前のCalculateの値を参照する.
		@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
		直前の計算結果を使用する */
	ErrorCode FeedforwardNeuralNetwork_CPU::Training(BATCH_BUFFER_POINTER o_lppDInputBuffer, CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer)
	{
		return  Training_device(&this->lpInputBuffer[0], o_lppDInputBuffer, NULL, i_lppDOutputBuffer);
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell

