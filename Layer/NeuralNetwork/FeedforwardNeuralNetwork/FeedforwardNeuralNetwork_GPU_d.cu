//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// GPU処理
//======================================
#include"stdafx.h"

#include<algorithm>

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU_d.cuh"

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
	FeedforwardNeuralNetwork_GPU_d::FeedforwardNeuralNetwork_GPU_d(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_inputDataStruct)
	{
	}
	/** コンストラクタ */
	FeedforwardNeuralNetwork_GPU_d::FeedforwardNeuralNetwork_GPU_d(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager)
		:	FeedforwardNeuralNetwork_GPU_base	(i_guid, i_layerData, i_inputDataStruct, i_temporaryMemoryManager)
	{
	}
	/** デストラクタ */
	FeedforwardNeuralNetwork_GPU_d::~FeedforwardNeuralNetwork_GPU_d()
	{
	}

	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 FeedforwardNeuralNetwork_GPU_d::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_GPU | Gravisbell::Layer::LAYER_KIND_DEVICEMEMORY;
	}


	//====================================
	// 出力バッファ関連
	//====================================
	/** 出力バッファの総数を設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU_d::SetOutputBufferCount(U32 i_outputBufferCount)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力バッファのサイズを設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU_d::ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize)
	{
		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 出力バッファの現在の使用者を取得する */
	GUID FeedforwardNeuralNetwork_GPU_d::GetReservedOutputBufferID(U32 i_outputBufferNo)
	{
		return GUID();
	}
	/** 出力バッファを使用中にして取得する(処理デバイス依存) */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU_d::ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid)
	{
		if(this->lpLayerOutputBuffer_d.count(i_guid) == 0)
		{
			// レイヤーの出力バッファを確保する
			IODataStruct outputDataStruct = this->GetOutputDataStruct(i_guid);

			this->lpLayerOutputBuffer_d[i_guid].resize(outputDataStruct.GetDataCount() * this->GetBatchSize());
		}

		return thrust::raw_pointer_cast(&this->lpLayerOutputBuffer_d[i_guid][0]);
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

