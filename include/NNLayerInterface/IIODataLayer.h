//=======================================
// 入出力信号データレイヤー
//=======================================
#ifndef __GRAVISBELL_I_IO_DATA_LAYER_H__
#define __GRAVISBELL_I_IO_DATA_LAYER_H__

#include"Common/Common.h"
#include"Common/ErrorCode.h"

#include"ISingleOutputLayer.h"
#include"ISingleInputLayer.h"
#include"IDataLayer.h"


namespace Gravisbell {
namespace NeuralNetwork {

	/** 入出力データレイヤー */
	class IIODataLayer : public ISingleOutputLayer, public ISingleInputLayer, public IDataLayer
	{
	public:
		/** コンストラクタ */
		IIODataLayer(){}
		/** デストラクタ */
		virtual ~IIODataLayer(){}

	public:
		/** 学習誤差を計算する.
			@param i_lppInputBuffer	入力データバッファ. [GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の要素数が必要
			直前の計算結果を使用する */
		virtual ErrorCode CalculateLearnError(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif