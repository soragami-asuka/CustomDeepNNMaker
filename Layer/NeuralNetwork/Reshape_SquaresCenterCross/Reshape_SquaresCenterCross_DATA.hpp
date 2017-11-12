/*--------------------------------------------
 * FileName  : Reshape_SquaresCenterCross_DATA.hpp
 * LayerName : 
 * guid      : 5C2729D1-33EB-45EF-ABA5-0C36AC22D0BC
 * 
 * Text      : X座標0を中心に入力信号を平方化する.
 *           : 出力X,Y信号数=sqrt(入力X信号数-1)×2+1
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_SquaresCenterCross_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_SquaresCenterCross_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Reshape_SquaresCenterCross {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ダミー
		  * ID   : dummy
		  */
		S32 dummy;

	};

} // Reshape_SquaresCenterCross
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Reshape_SquaresCenterCross_H__
