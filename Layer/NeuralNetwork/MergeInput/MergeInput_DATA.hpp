/*--------------------------------------------
 * FileName  : MergeInput_DATA.hpp
 * LayerName : 入力結合レイヤー
 * guid      : 53DAEC93-DBDB-4048-BD5A-401DD005C74E
 * 
 * Text      : 入力信号を結合して出力する
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeInput_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeInput_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace MergeInput {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : 結合方向
		  * ID   : mergeDirection
		  * Text : どの次元を使用して結合を行うかの設定.
		  *       : 指定された次元以外の値は全て同じサイズである必要がある.
		  */
		enum : S32{
			/** Name : X
			  * ID   : x
			  * Text : X軸
			  *      : (null)
			  */
			mergeDirection_x,

			/** Name : Y
			  * ID   : y
			  * Text : Y軸
			  *      : (null)
			  */
			mergeDirection_y,

			/** Name : Z
			  * ID   : z
			  * Text : Z軸
			  *      : 
			  */
			mergeDirection_z,

			/** Name : CH
			  * ID   : ch
			  * Text : CH
			  *      : (null)
			  */
			mergeDirection_ch,

		}mergeDirection;

	};

} // MergeInput
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_MergeInput_H__
