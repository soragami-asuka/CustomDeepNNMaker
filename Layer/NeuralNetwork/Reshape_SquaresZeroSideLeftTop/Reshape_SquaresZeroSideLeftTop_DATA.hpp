/*--------------------------------------------
 * FileName  : Reshape_SquaresZeroSideLeftTop_DATA.hpp
 * LayerName : 
 * guid      : F6D9C5DA-D583-455B-9254-5AEF3CA9021B
 * 
 * Text      : X座標0を中心に入力信号を平方化する.
 *           : X×Y+1の入力信号を必要とする
 *           : X=0 or Y=0を元データのX=0とする
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_SquaresZeroSideLeftTop_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_SquaresZeroSideLeftTop_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Reshape_SquaresZeroSideLeftTop {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : X
		  * ID   : x
		  * Text : 出力構造のXサイズ
		  */
		S32 x;

		/** Name : Y
		  * ID   : y
		  * Text : 出力構造のYサイズ
		  */
		S32 y;

	};

} // Reshape_SquaresZeroSideLeftTop
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Reshape_SquaresZeroSideLeftTop_H__
