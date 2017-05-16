//=======================================
// 複数入力を持つNNのレイヤー
//=======================================
#ifndef __GRAVISBELL_I_NN_MULT_INPUT_LAYER_H__
#define __GRAVISBELL_I_NN_MULT_INPUT_LAYER_H__

#include"../IO/IMultInputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNMultInputLayer : public IO::IMultInputLayer
	{
	public:
		/** コンストラクタ */
		INNMultInputLayer() : IMultInputLayer(){}
		/** デストラクタ */
		virtual ~INNMultInputLayer(){}

	public:
		/** 演算処理を実行する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif