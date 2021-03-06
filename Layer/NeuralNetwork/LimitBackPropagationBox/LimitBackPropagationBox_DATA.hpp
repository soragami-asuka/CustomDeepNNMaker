/*--------------------------------------------
 * FileName  : LimitBackPropagationBox_DATA.hpp
 * LayerName : ãû`À·éÍÍð§À·é
 * guid      : 89359466-E1B2-4129-90E8-50C74E4BC597
 * 
 * Text      : ãû`À·éÍÍð§À·é.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_LimitBackPropagationBox_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_LimitBackPropagationBox_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace LimitBackPropagationBox {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : JnÊu
		  * ID   : startPosition
		  */
		Vector3D<S32> startPosition;

		/** Name : oXYZTCY
		  * ID   : boxSize
		  */
		Vector3D<S32> boxSize;

	};

} // LimitBackPropagationBox
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_LimitBackPropagationBox_H__
