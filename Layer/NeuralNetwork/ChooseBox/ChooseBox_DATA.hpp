/*--------------------------------------------
 * FileName  : ChooseBox_DATA.hpp
 * LayerName : 入力信号から特定XYZ区間を抽出する
 * guid      : 14086A2C-2B99-4849-BC19-B2238BBDA5B7
 * 
 * Text      : 入力信号から特定XYZ区間を抽出する.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ChooseBox_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ChooseBox_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace ChooseBox {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 開始位置
		  * ID   : startPosition
		  */
		Vector3D<S32> startPosition;

		/** Name : 抽出XYZサイズ
		  * ID   : boxSize
		  */
		Vector3D<S32> boxSize;

	};

} // ChooseBox
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ChooseBox_H__
