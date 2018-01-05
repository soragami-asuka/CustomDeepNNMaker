//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// GPU処理
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_HOST_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_GPU_HOST_H__

#include"FeedforwardNeuralNetwork_GPU_base.cuh"

#include<Layer/NeuralNetwork/ILayerDLLManager.h>

#include<thrust/device_vector.h>

#include"../_LayerBase/CLayerBase_GPU.cuh"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_GPU_h : public FeedforwardNeuralNetwork_GPU_base
	{
	private:
		struct BufferInfo
		{
			GUID reserveLayerID;
			thrust::device_vector<F32> lpBuffer;
		};

		std::map<GUID, std::vector<F32>>	lpLayerOutputBuffer_h;	/**< 各レイヤーごとの出力バッファ(ホストメモリ) */
		std::vector<BufferInfo> lpTemporaryOutputBuffer;

		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_GPU_h(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct);
		/** コンストラクタ */
		FeedforwardNeuralNetwork_GPU_h(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData, const IODataStruct& i_inputDataStruct, Gravisbell::Common::ITemporaryMemoryManager& i_temporaryMemoryManager);
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_GPU_h();

	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKind(void)const;



		//====================================
		// 出力バッファ関連
		//====================================
	protected:
		/** 出力バッファの総数を設定する */
		ErrorCode SetOutputBufferCount(U32 i_outputBufferCount);

		/** 出力バッファのサイズを設定する */
		ErrorCode ResizeOutputBuffer(U32 i_outputBufferNo, U32 i_bufferSize);

	public:
		/** 出力バッファの現在の使用者を取得する */
		GUID GetReservedOutputBufferID(U32 i_outputBufferNo);
		/** 出力バッファを使用中にして取得する(処理デバイス依存) */
		BATCH_BUFFER_POINTER ReserveOutputBuffer_d(U32 i_outputBufferNo, GUID i_guid);

	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif