/*--------------------------------------------
 * FileName  : Activation_DATA.hpp
 * LayerName : 活性化関数
 * guid      : 99904134-83B7-4502-A0CA-728A2C9D80C7
 * 
 * Text      : 活性化関数
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Activation_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Activation_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Activation {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 活性化関数種別
		  * ID   : ActivationType
		  * Text : 使用する活性化関数の種類を定義する
		  */
		enum : S32{
			/** Name : リニア関数
			  * ID   : lenear
			  * Text : y = x;
			  */
			ActivationType_lenear,

			/** Name : シグモイド関数
			  * ID   : sigmoid
			  * Text : y = 1 / (1 + e^(-x));
			  *      : 範囲 0 < y < 1
			  *      : (x=0, y=0.5)を通る
			  */
			ActivationType_sigmoid,

			/** Name : シグモイド関数(出力レイヤー用)
			  * ID   : sigmoid_crossEntropy
			  * Text : y = 1 / (1 + e^(-x));
			  *      : 範囲 0 < y < 1
			  *      : (x=0, y=0.5)を通る
			  */
			ActivationType_sigmoid_crossEntropy,

			/** Name : ReLU（ランプ関数）
			  * ID   : ReLU
			  * Text : y = max(0, x);
			  *      : 範囲 0 <= y
			  *      : (x=0, y=0)を通る
			  */
			ActivationType_ReLU,

			/** Name : Leaky-ReLU
			  * ID   : LeakyReLU
			  * Text : y = max(alpha*x, x);
			  *      : (x=0, y=0)を通る
			  */
			ActivationType_LeakyReLU,

			/** Name : tanh(双曲線関数)
			  * ID   : tanh
			  * Text : y = sin(x)/cos(x);
			  */
			ActivationType_tanh,

			/** Name : SoftMax関数
			  * ID   : softmax_ALL
			  * Text : 全体における自身の割合を返す関数.
			  *      : y = e^x / Σe^x;
			  */
			ActivationType_softmax_ALL,

			/** Name : SoftMax関数(出力レイヤー用)
			  * ID   : softmax_ALL_crossEntropy
			  * Text : 全体における自身の割合を返す関数.
			  *      : y = e^x / Σe^x;
			  */
			ActivationType_softmax_ALL_crossEntropy,

			/** Name : SoftMax関数(CH内のみ)
			  * ID   : softmax_CH
			  * Text : 同一のX,Y,Zにおける各CHの自身の割合を返す関数.
			  *      : y = e^x / Σe^x;
			  */
			ActivationType_softmax_CH,

			/** Name : SoftMax関数(CH内のみ)(出力レイヤー用)
			  * ID   : softmax_CH_crossEntropy
			  * Text : 同一のX,Y,Zにおける各CHの自身の割合を返す関数.
			  *      : y = e^x / Σe^x;
			  */
			ActivationType_softmax_CH_crossEntropy,

		}ActivationType;

		/** Name : Leaky-ReLU-Alpha
		  * ID   : LeakyReLU_alpha
		  * Text : Leaky-ReLUで使用するαの値
		  */
		F32 LeakyReLU_alpha;

	};

} // Activation
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Activation_H__
