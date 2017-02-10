//=======================================
// 入出力信号データレイヤー
//=======================================
#ifndef __I_INPUT_DATA_LAYER_H__
#define __I_INPUT_DATA_LAYER_H__

#include"ISingleOutputLayer.h"
#include"ISingleInputLayer.h"
#include"IDataLayer.h"

namespace CustomDeepNNLibrary
{
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
		virtual ELayerErrorCode CalculateLearnError(const float** i_lppInputBuffer) = 0;

	};
}

#endif