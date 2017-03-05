//=======================================
// レイヤーベース
//=======================================
#ifndef __GRAVISBELL_I_INPUT_LAYER_H__
#define __GRAVISBELL_I_INPUT_LAYER_H__

#include"Common/ErrorCode.h"
#include"Common/IODataStruct.h"

#include"ILayerBase.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** レイヤーベース */
	class IInputLayer : public virtual ILayerBase
	{
	public:
		/** コンストラクタ */
		IInputLayer(){}
		/** デストラクタ */
		virtual ~IInputLayer(){}
	};

}	// NeuralNetwork
}	// Gravisbell

#endif