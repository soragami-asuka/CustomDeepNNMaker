//======================================
// フィードフォワードニューラルネットワークの処理レイヤー
// 複数のレイヤーを内包し、処理する
// CPU処理
//======================================
#ifndef __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_CPU_H__
#define __GRAVISBELL_FEEDFORWARD_NEURALNETWORK_CPU_H__

#include"FeedforwardNeuralNetwork_Base.h"

#include<Layer/NeuralNetwork/ILayerDLLManager.h>


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class FeedforwardNeuralNetwork_CPU : public FeedforwardNeuralNetwork_Base
	{
		//====================================
		// コンストラクタ/デストラクタ
		//====================================
	public:
		/** コンストラクタ */
		FeedforwardNeuralNetwork_CPU(const Gravisbell::GUID& i_guid, class FeedforwardNeuralNetwork_LayerData_Base& i_layerData);
		/** デストラクタ */
		virtual ~FeedforwardNeuralNetwork_CPU();

	public:
		/** レイヤー種別の取得.
			ELayerKind の組み合わせ. */
		U32 GetLayerKind(void)const;

	public:
		//====================================
		// 入出力バッファ関連
		//====================================
		/** 出力データバッファを取得する.
			@param o_lpOutputBuffer	出力データ格納先配列. [GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0 */
		ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const;

		/** 学習差分を取得する.
			@param lpDInputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		ErrorCode GetDInputBuffer(BATCH_BUFFER_POINTER o_lpDInputBuffer)const;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif