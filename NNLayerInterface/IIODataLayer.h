//=======================================
// 入出力信号データレイヤー
//=======================================
#ifndef __I_INPUT_DATA_LAYER_H__
#define __I_INPUT_DATA_LAYER_H__

#include"IOutputLayer.h"
#include"IInputLayer.h"
#include"IDataLayer.h"

namespace CustomDeepNNLibrary
{
	/** 入出力データレイヤー */
	class IIODataLayer : public IOutputLayer, public IInputLayer, public IDataLayer
	{
	public:
		/** コンストラクタ */
		IIODataLayer(){}
		/** デストラクタ */
		virtual ~IIODataLayer(){}

	};
}

#endif