//=======================================
// 単一入力を持つNNのレイヤー
//=======================================
#ifndef __GRAVISBELL_I_NN_SINGLE_INPUT_LAYER_H__
#define __GRAVISBELL_I_NN_SINGLE_INPUT_LAYER_H__

#include"../IO/ISingleInputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNSingleInputLayer : public IO::ISingleInputLayer
	{
	public:
		/** コンストラクタ */
		INNSingleInputLayer() : ISingleInputLayer(){}
		/** デストラクタ */
		virtual ~INNSingleInputLayer(){}

	public:
		/** 演算処理を実行する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif