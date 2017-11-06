/*--------------------------------------------
 * FileName  : Reshape_DATA.hpp
 * LayerName : ���͐M���\����ϊ�����
 * guid      : E78E7F59-D4B3-45A1-AEEB-9F2A5155473F
 * 
 * Text      : ���͐M���\����ϊ�����.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Reshape {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : X���v�f��
		  * ID   : x
		  */
		S32 x;

		/** Name : Y���v�f��
		  * ID   : y
		  */
		S32 y;

		/** Name : Z���v�f��
		  * ID   : z
		  */
		S32 z;

		/** Name : CH�v�f��
		  * ID   : ch
		  */
		S32 ch;

	};

} // Reshape
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Reshape_H__
