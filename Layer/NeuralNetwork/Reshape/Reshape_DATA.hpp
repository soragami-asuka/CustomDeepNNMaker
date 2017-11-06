/*--------------------------------------------
 * FileName  : Reshape_DATA.hpp
 * LayerName : 入力信号構造を変換する
 * guid      : E78E7F59-D4B3-45A1-AEEB-9F2A5155473F
 * 
 * Text      : 入力信号構造を変換する.
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
		/** Name : X軸要素数
		  * ID   : x
		  */
		S32 x;

		/** Name : Y軸要素数
		  * ID   : y
		  */
		S32 y;

		/** Name : Z軸要素数
		  * ID   : z
		  */
		S32 z;

		/** Name : CH要素数
		  * ID   : ch
		  */
		S32 ch;

	};

} // Reshape
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Reshape_H__
