//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// GPU処理
//======================================
#include"stdafx.h"

#include<algorithm>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU_h.cuh"

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
	FeedforwardNeuralNetwork_GPU_h::FeedforwardNeuralNetwork_GPU_h(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_lpInputDataStruct, i_inputLayerCount)
	{
	}
	/** コンストラクタ */
	FeedforwardNeuralNetwork_GPU_h::FeedforwardNeuralNetwork_GPU_h(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct i_lpInputDataStruct[], U32 i_inputLayerCount, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_lpInputDataStruct, i_inputLayerCount, i_temporaryMemoryManager)
	{
	}
	/** デストラクタ */
	FeedforwardNeuralNetwork_GPU_h::~FeedforwardNeuralNetwork_GPU_h()
	{
	}

	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 FeedforwardNeuralNetwork_GPU_h::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_GPU | Gravisbell::Layer::LAYER_KIND_HOSTMEMORY;
	}


	//====================================
	// 出力バッファ関連
	//====================================
	/** 出力バッファの総数を設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU_h::SetOutputBufferCount(U32 i_outputBufferCount)
	{
		this->lpTemporaryOutputBuffer.resize(i_outputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力バッファのサイズを設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU_h::ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize)
	{
		if(i_outputBufferNo >= this->lpTemporaryOutputBuffer.size())
			return ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer.resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力バッファの現在の使用者を取得する */
	GUID FeedforwardNeuralNetwork_GPU_h::GetReservedOutputBufferID(U32 i_outputBufferNo)
	{
		if(i_outputBufferNo >= this->lpTemporaryOutputBuffer.size())
			return GUID();

		return this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID;
	}
	/** 出力バッファを使用中にして取得する(処理デバイス依存) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_h::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid)
	{
		if(i_outputBufferNo >= this->lpTemporaryOutputBuffer.size())
			return NULL;

		if(this->lpLayerOutputBuffer_h.count(i_guid) == 0)
		{
			// レイヤーの出力バッファを確保する
			IODataStruct outputDataStruct = this->GetOutputDataStruct(i_guid);

			this->lpLayerOutputBuffer_h[i_guid].resize(outputDataStruct.GetDataCount() * this->GetBatchSize());
		}

		// 対象のバッファを予約中でない場合、現在のバッファをホスト側に退避.新しいバッファの内容をデバイス側にコピー
		if(this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID != i_guid)
		{
			// 退避
			{
				auto it_buffer = this->lpLayerOutputBuffer_h.find(this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID);
				if(it_buffer != this->lpLayerOutputBuffer_h.end())
				{
					cudaMemcpy(
						&it_buffer->second[0],
						thrust::raw_pointer_cast(&this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer[0]),
						sizeof(F32) * it_buffer->second.size(),
						cudaMemcpyDeviceToHost);
				}
			}

			// データコピー
			{
				auto it_buffer = this->lpLayerOutputBuffer_h.find(i_guid);
				if(it_buffer != this->lpLayerOutputBuffer_h.end())
				{
					cudaMemcpy(
						thrust::raw_pointer_cast(&this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer[0]),
						&it_buffer->second[0],
						sizeof(F32) * it_buffer->second.size(),
						cudaMemcpyHostToDevice);
				}
			}
		}

		// 予約情報を更新
		this->lpTemporaryOutputBuffer[i_outputBufferNo].reserveLayerID = i_guid;

		return thrust::raw_pointer_cast(&this->lpTemporaryOutputBuffer[i_outputBufferNo].lpBuffer[0]);
	}



}	// NeuralNetwork
}	// Layer
}	// Gravisbell

