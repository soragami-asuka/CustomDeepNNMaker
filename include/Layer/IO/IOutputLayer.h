//=======================================
// レイヤーベース
//=======================================
#ifndef __GRAVISBELL_I_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_OUTPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** レイヤーベース */
	class IOutputLayer : public virtual ILayerBase
	{
	public:
		/** コンストラクタ */
		IOutputLayer(){}
		/** デストラクタ */
		virtual ~IOutputLayer(){}
	};

}	// IO
}	// Layer
}	// GravisBell

#endif