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

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : 最適化ルーチン
		  * ID   : Optimizer
		  * Text : 重み誤差を反映させるためのアルゴリズム.
		  */
		enum : S32{
			/** Name : SGD
			  * ID   : SGD
			  */
			Optimizer_SGD,

			/** Name : Momentum
			  * ID   : Momentum
			  * Text : 慣性付与
			  */
			Optimizer_Momentum,

			/** Name : AdaDelta
			  * ID   : AdaDelta
			  */
			Optimizer_AdaDelta,

			/** Name : Adam
			  * ID   : Adam
			  */
			Optimizer_Adam,

		}Optimizer;

		/** Name : 学習係数
		  * ID   : LearnCoeff
		  * Text : SGD,Momentumで使用.
		  */
		F32 LearnCoeff;

		/** Name : Momentum-α
		  * ID   : Momentum_alpha
		  * Text : Momentumで使用.t-1の値を反映する割合
		  */
		F32 Momentum_alpha;

		/** Name : AdaDelta-Ρ
		  * ID   : AdaDelta_rho
		  * Text : AdaDeltaで使用.減衰率.高いほうが減衰しづらい.
		  */
		F32 AdaDelta_rho;

		/** Name : AdaDelta-ε
		  * ID   : AdaDelta_epsilon
		  * Text : AdaDeltaで使用.補助数.高いほど初期値が大きくなる.
		  */
		F32 AdaDelta_epsilon;

		/** Name : Adam-α
		  * ID   : Adam_alpha
		  * Text : Adamで使用.加算率.高いほうが更新されやすい.
		  */
		F32 Adam_alpha;

		/** Name : Adam-β
		  * ID   : Adam_beta1
		  * Text : Adamで使用.減衰率1.高いほうが減衰しづらい.
		  */
		F32 Adam_beta1;

		/** Name : Adam-β
		  * ID   : Adam_beta2
		  * Text : Adamで使用.減衰率2.高いほうが減衰しづらい.
		  */
		F32 Adam_beta2;

		/** Name : Adam-ε
		  * ID   : Adam_epsilon
		  * Text : AdaDeltaで使用.補助数.高いほど初期値が大きくなる.
		  */
		F32 Adam_epsilon;

	};

} // BatchNormalization
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_BatchNormalization_H__
