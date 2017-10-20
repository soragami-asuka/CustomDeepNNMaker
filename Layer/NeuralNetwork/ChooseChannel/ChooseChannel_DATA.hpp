/*--------------------------------------------
 * FileName  : ChooseChannel_DATA.hpp
 * LayerName : 入力信号から特定チャンネルを抽出する
 * guid      : 244824B3-BCFC-4655-A991-0F6136D37A34
 * 
 * Text      : 入力信号から特定チャンネルを抽出する.
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
		/** Name : 開始チャンネル番号
		  * ID   : startChannelNo
		  */
		S32 startChannelNo;

		/** Name : 出力チャンネル数
		  * ID   : channelCount
		  */
		S32 channelCount;

	};

} // ChooseChannel
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ChooseChannel_H__
