/*--------------------------------------------
 * FileName  : BatchNormalizationAll_DATA.hpp
 * LayerName : バッチ正規化(チャンネル区別なし)
 * guid      : 8AECB925-8DCF-4876-BA6A-6ADBE280D285
 * 
 * Text      : チャンネルの区別なくバッチ単位で正規化を行う.(≒ 1chのデータとして扱う)
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalizationAll_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_BatchNormalizationAll_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace BatchNormalizationAll {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 安定化係数
		  * ID   : epsilon
		  * Text : 分散の値が小さすぎる場合に割り算を安定させるための値
		  */
		F32 epsilon;

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

} // BatchNormalizationAll
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalizationAll_H__
