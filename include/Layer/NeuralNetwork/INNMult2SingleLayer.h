//=======================================
// 複数入力、単一出力を持つNNのレイヤー
//=======================================
#ifndef __GRAVISBELL_I_NN_MULT_TO_SINGLE_LAYER_H__
#define __GRAVISBELL_I_NN_MULT_TO_SINGLE_LAYER_H__

#include"../IO/IMultInputLayer.h"
#include"../IO/ISingleOutputLayer.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNMult2SingleLayer : public IO::IMultInputLayer, public IO::ISingleOutputLayer
	{
	public:
		/** コンストラクタ */
		INNMult2SingleLayer() : IMultInputLayer(), ISingleOutputLayer(){}
		/** デストラクタ */
		virtual ~INNMult2SingleLayer(){}

	public:
		/** 演算処理を実行する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]) = 0;

		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif