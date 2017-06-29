//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// GPU処理
//======================================
#include"stdafx.h"

#include"FeedforwardNeuralNetwork_Base.h"
#include"FeedforwardNeuralNetwork_GPU.cuh"

#include"FeedforwardNeuralNetwork_LayerData_Base.h"

// CUDA用
#pragma warning(push)
#pragma warning(disable : 4267)
#include <cuda_runtime_api.h>
#pragma warning(pop)


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** コンストラクタ */
	FeedforwardNeuralNetwork_GPU::FeedforwardNeuralNetwork_GPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct)
		:	FeedforwardNeuralNetwork_Base	(i_guid, i_layerData, i_inputDataStruct, i_layerData.GetOutputDataStruct(&i_inputDataStruct, 1))
	{
	}
	/** デストラクタ */
	FeedforwardNeuralNetwork_GPU::~FeedforwardNeuralNetwork_GPU()
	{
	}

	/** レイヤー種別の取得.
		ELayerKind の組み合わせ. */
	U32 FeedforwardNeuralNetwork_GPU::GetLayerKind(void)const
	{
		return this->GetLayerKindBase() | Gravisbell::Layer::LAYER_KIND_GPU;
	}


	//====================================
	// 入力誤差バッファ関連
	//====================================
	/** 入力誤差バッファの総数を設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU::SetDInputBufferCount(U32 i_DInputBufferCount)
	{
		this->lpDInputBuffer.resize(i_DInputBufferCount);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 入力誤差バッファのサイズを設定する */
	ErrorCode FeedforwardNeuralNetwork_GPU::ResizeDInputBuffer(U32 i_DInputBufferNo, U32 i_bufferSize)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			ErrorCode::ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE;

		this->lpDInputBuffer[i_DInputBufferNo].resize(i_bufferSize);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 入力誤差バッファを取得する */
	BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU::GetDInputBuffer(U32 i_DInputBufferNo)
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_DInputBufferNo][0]);
	}
	/** 入力誤差バッファを取得する */
	CONST_BATCH_BUFFER_POINTER FeedforwardNeuralNetwork_GPU::GetDInputBuffer(U32 i_DInputBufferNo)const
	{
		if(i_DInputBufferNo >= this->lpDInputBuffer.size())
			return NULL;

		return thrust::raw_pointer_cast(&this->lpDInputBuffer[i_DInputBufferNo][0]);
	}


	//====================================
	// 入出力バッファ関連
	//====================================
	/** 出力データバッファを取得する.
		@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
		@return 成功した場合0 */
	ErrorCode FeedforwardNeuralNetwork_GPU::GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const
	{
		if(o_lpOutputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 outputBufferCount = this->GetOutputBufferCount();

		cudaMemcpy(o_lpOutputBuffer, FeedforwardNeuralNetwork_Base::GetOutputBuffer(), sizeof(F32)*outputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}

	/** 学習差分を取得する.
		@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
	ErrorCode FeedforwardNeuralNetwork_GPU::GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const
	{
		if(o_lpDInputBuffer == NULL)
			return ErrorCode::ERROR_CODE_COMMON_NULL_REFERENCE;

		const U32 batchSize = this->GetBatchSize();
		const U32 inputBufferCount = this->GetInputBufferCount();

		cudaMemcpy(o_lpDInputBuffer, FeedforwardNeuralNetwork_Base::GetDInputBuffer(), sizeof(F32)*inputBufferCount*batchSize, cudaMemcpyDeviceToHost);

		return ErrorCode::ERROR_CODE_NONE;
	}


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

