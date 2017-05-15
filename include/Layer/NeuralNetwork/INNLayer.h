//=======================================
// レイヤーベース
//=======================================
#ifndef __GRAVISBELL_I_NN_LAYER_BASE_H__
#define __GRAVISBELL_I_NN_LAYER_BASE_H__

#include"../IO/ISingleInputLayer.h"
#include"../IO/ISingleOutputLayer.h"

#include"../../SettingData/Standard/IData.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーベース */
	class INNLayer : public IO::ISingleInputLayer, public virtual IO::ISingleOutputLayer
	{
	public:
		/** コンストラクタ */
		INNLayer(){}
		/** デストラクタ */
		virtual ~INNLayer(){}

	public:


	public:
		/** 演算処理を実行する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			@return 成功した場合0が返る */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;

		/** 学習処理を実行する.
			入力信号、出力信号は直前のCalculateの値を参照する.
			@param	i_lppDOutputBuffer	出力誤差差分=次レイヤーの入力誤差差分.	[GetBatchSize()の戻り値][GetOutputBufferCount()の戻り値]の要素数が必要.
			直前の計算結果を使用する */
		virtual ErrorCode Training(CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif