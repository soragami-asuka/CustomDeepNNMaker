/*--------------------------------------------
 * FileName  : SeparateOutput_DATA.hpp
 * LayerName : 出力信号分割レイヤー
 * guid      : C13C30DA-056E-46D0-90FC-608766FB432E
 * 
 * Text      : 出力信号を分割するレイヤー.
 *           : 各分割内での出力信号数に変化はないが、誤差のマージを行うことができる.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SeparateOutput_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SeparateOutput_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace SeparateOutput {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 出力信号の分割数
		  * ID   : separateCount
		  * Text : 出力信号を何個に分割するか
		  */
		S32 separateCount;

	};

} // SeparateOutput
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SeparateOutput_H__
