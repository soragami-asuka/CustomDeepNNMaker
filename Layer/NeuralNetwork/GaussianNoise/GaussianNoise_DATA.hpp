/*--------------------------------------------
 * FileName  : GaussianNoise_DATA.hpp
 * LayerName : GaussianNoise
 * guid      : AC27C912-A11D-4519-81A0-17C078E4431F
 * 
 * Text      : GaussianNoise.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_GaussianNoise_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_GaussianNoise_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace GaussianNoise {

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : 平均
		  * ID   : GaussianNoise_Average
		  * Text : 発生するノイズの平均値.ノイズのバイアス
		  */
		F32 GaussianNoise_Average;

		/** Name : 分散
		  * ID   : GaussianNoise_Variance
		  * Text : 発生するノイズの分散.ノイズの強度
		  */
		F32 GaussianNoise_Variance;

	};

} // GaussianNoise
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_GaussianNoise_H__
