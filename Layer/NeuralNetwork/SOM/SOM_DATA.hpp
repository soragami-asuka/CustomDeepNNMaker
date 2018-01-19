/*--------------------------------------------
 * FileName  : SOM_DATA.hpp
 * LayerName : 自己組織化マップ
 * guid      : AF36DF4D-9F50-46FF-A1C1-5311CA761F6A
 * 
 * Text      : 自己組織化マップ.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SOM_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SOM_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace SOM {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 入力バッファ数
		  * ID   : InputBufferCount
		  * Text : レイヤーに対する入力バッファ数
		  */
		S32 InputBufferCount;

		/** Name : 次元数
		  * ID   : DimensionCount
		  * Text : 生成されるマップの次元数
		  */
		S32 DimensionCount;

		/** Name : 分解能
		  * ID   : ResolutionCount
		  * Text : 次元ごとの分解性能
		  */
		S32 ResolutionCount;

		/** Name : 初期化最小値
		  * ID   : InitializeMinValue
		  * Text : 初期化に使用する値の最小値
		  */
		F32 InitializeMinValue;

		/** Name : 初期化最大値
		  * ID   : InitializeMaxValue
		  * Text : 初期化に使用する値の最大値
		  */
		F32 InitializeMaxValue;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : 学習係数
		  * ID   : SOM_L0
		  * Text : パラメータ更新の係数
		  */
		F32 SOM_L0;

		/** Name : 時間減衰率
		  * ID   : SOM_ramda
		  * Text : 学習回数に応じた学習率の減衰率.値が高いほうが減衰率は低い
		  */
		F32 SOM_ramda;

		/** Name : 距離減衰率
		  * ID   : SOM_sigma
		  * Text : 更新個体とBMUとの距離に応じた減衰率.値が高いほうが減衰率は低い
		  */
		F32 SOM_sigma;

	};

} // SOM
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SOM_H__
