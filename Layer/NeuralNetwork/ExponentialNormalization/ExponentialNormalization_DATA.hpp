/*--------------------------------------------
 * FileName  : ExponentialNormalization_DATA.hpp
 * LayerName : バッチ正規化
 * guid      : 44F733E8-417C-4598-BF05-2CC26E1AB6F1
 * 
 * Text      : バッチ単位で正規化を行う
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ExponentialNormalization_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ExponentialNormalization_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace ExponentialNormalization {

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

		/** Name : 平滑化時間数
		  * ID   : ExponentialTime
		  * Text : 平滑化時間数
		  */
		S32 ExponentialTime;

		/** Name : 初期化時間数
		  * ID   : InitParameterTime
		  * Text : 初期化時間数.初期化するために使用する時間数.
		  */
		S32 InitParameterTime;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : 加速係数
		  * ID   : AccelCoeff
		  * Text : 加速係数
		  */
		F32 AccelCoeff;

	};

} // ExponentialNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ExponentialNormalization_H__
