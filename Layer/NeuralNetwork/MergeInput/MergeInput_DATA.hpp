/*--------------------------------------------
 * FileName  : MergeInput_DATA.hpp
 * LayerName : ���͌������C���[
 * guid      : 53DAEC93-DBDB-4048-BD5A-401DD005C74E
 * 
 * Text      : ���͐M�����������ďo�͂���
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
		/** Name : ��������
		  * ID   : mergeDirection
		  * Text : �ǂ̎������g�p���Č������s�����̐ݒ�.
		  *       : �w�肳�ꂽ�����ȊO�̒l�͑S�ē����T�C�Y�ł���K�v������.
		  */
		enum : S32{
			/** Name : X
			  * ID   : x
			  * Text : X��
			  *      : (null)
			  */
			mergeDirection_x,

			/** Name : Y
			  * ID   : y
			  * Text : Y��
			  *      : (null)
			  */
			mergeDirection_y,

			/** Name : Z
			  * ID   : z
			  * Text : Z��
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
