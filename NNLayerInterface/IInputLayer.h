//=======================================
// レイヤーベース
//=======================================
#ifndef __I_INPUT_LAYER_H__
#define __I_INPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** レイヤーベース */
	class IInputLayer : public virtual ILayerBase
	{
	public:
		/** コンストラクタ */
		IInputLayer(){}
		/** デストラクタ */
		virtual ~IInputLayer(){}
	};
}

#endif