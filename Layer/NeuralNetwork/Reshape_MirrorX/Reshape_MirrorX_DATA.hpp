/*--------------------------------------------
 * FileName  : Reshape_MirrorX_DATA.hpp
 * LayerName : 
 * guid      : DFCA3F81-C2F1-4AC6-B618-816651ADDB63
 * 
 * Text      : X座標0を中心に入力信号をミラー化する.
 *           : 出力X信号数=入力X信号数×2-1
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_MirrorX_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_MirrorX_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Reshape_MirrorX {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ダミー
		  * ID   : dummy
		  */
		S32 dummy;

	};

} // Reshape_MirrorX
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Reshape_MirrorX_H__
