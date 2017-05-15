//=======================================
// 複数出力を持つNNのレイヤー
//=======================================
#ifndef __GRAVISBELL_I_NN_MULT_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_NN_MULT_OUTPUT_LAYER_H__

#include"../IO/IMultOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNMultOutputLayer : public IO::IMultOutputLayer
	{
	public:
		/** コンストラクタ */
		INNMultOutputLayer(){}
		/** デストラクタ */
		virtual ~INNMultOutputLayer(){}


	public:
		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer[]	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要な配列の[GetOutputDataCount()]配列
			直前の計算結果を使用する */
		virtual ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer[]) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif