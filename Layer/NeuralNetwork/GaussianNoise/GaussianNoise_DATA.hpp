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

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ����
		  * ID   : Average
		  * Text : ��������m�C�Y�̕��ϒl.�m�C�Y�̃o�C�A�X
		  */
		F32 Average;

		/** Name : ���U
		  * ID   : Variance
		  * Text : ��������m�C�Y�̕��U.�m�C�Y�̋��x
		  */
		F32 Variance;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : ����
		  * ID   : GaussianNoise_Bias
		  * Text : ��������m�C�Y�̕��ϒl.�m�C�Y�̃o�C�A�X
		  */
		F32 GaussianNoise_Bias;

		/** Name : ���U
		  * ID   : GaussianNoise_Power
		  * Text : �m�C�Y�̋��x
		  */
		F32 GaussianNoise_Power;

	};

} // GaussianNoise
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_GaussianNoise_H__
