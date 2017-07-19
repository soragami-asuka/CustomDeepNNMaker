/*--------------------------------------------
 * FileName  : BatchNormalization_DATA.hpp
 * LayerName : バッチ正規化
 * guid      : ACD11A5A-BFB5-4951-8382-1DE89DFA96A8
 * 
 * Text      : バッチ単位で正規化を行う
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalization_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalization_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace BatchNormalization {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 安定化係数
		  * ID   : epsilon
		  * Text : 分散の値が小さすぎる場合に割り算を安定させるための値
		  */
		F32 epsilon;

		/** Name : 入力チャンネル数
		  * ID   : InputChannelCount
		  * Text : 入力チャンネル数
		  */
		S32 InputChannelCount;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : 最小平均値更新係数
		  * ID   : AverageUpdateCoeffMin
		  * Text : 平均値を更新する際の係数の最小値.0=Epochの全データの平均値を使用する.1=直近のデータの平均値を使用する.
		  */
		F32 AverageUpdateCoeffMin;

	};

} // BatchNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalization_H__
