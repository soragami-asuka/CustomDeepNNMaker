//=======================================
// レイヤーベース
//=======================================
#ifndef __I_OUTPUT_LAYER_H__
#define __I_OUTPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** レイヤーベース */
	class IOutputLayer : public virtual ILayerBase
	{
	public:
		/** コンストラクタ */
		IOutputLayer(){}
		/** デストラクタ */
		virtual ~IOutputLayer(){}
	};
}

#endif