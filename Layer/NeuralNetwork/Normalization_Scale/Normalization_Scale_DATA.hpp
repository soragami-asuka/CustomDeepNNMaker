/*--------------------------------------------
 * FileName  : Normalization_Scale_DATA.hpp
 * LayerName : スケール値で正規化
 * guid      : D8C0DE15-5445-482D-BBC9-0026BFA96ADD
 * 
 * Text      : 全てのチャンネルをスケール値と平均値を用いて正規化する
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Normalization_Scale_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Normalization_Scale_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Normalization_Scale {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : dummy
		  * ID   : dummy
		  * Text : dummy
		  */
		S32 dummy;

	};

} // Normalization_Scale
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Normalization_Scale_H__
