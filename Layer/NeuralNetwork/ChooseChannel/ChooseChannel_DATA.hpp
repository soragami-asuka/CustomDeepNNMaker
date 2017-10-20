/*--------------------------------------------
 * FileName  : ChooseChannel_DATA.hpp
 * LayerName : ���͐M���������`�����l���𒊏o����
 * guid      : 244824B3-BCFC-4655-A991-0F6136D37A34
 * 
 * Text      : ���͐M���������`�����l���𒊏o����.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ChooseChannel_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ChooseChannel_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace ChooseChannel {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �J�n�`�����l���ԍ�
		  * ID   : startChannelNo
		  */
		S32 startChannelNo;

		/** Name : �o�̓`�����l����
		  * ID   : channelCount
		  */
		S32 channelCount;

	};

} // ChooseChannel
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ChooseChannel_H__
