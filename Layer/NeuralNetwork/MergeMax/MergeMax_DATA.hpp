/*--------------------------------------------
 * FileName  : MergeMax_DATA.hpp
 * LayerName : ���C���[�̃}�[�W(�ő�l)
 * guid      : 3F015946-7E88-4DB0-91BD-F4013F2190D4
 * 
 * Text      : ���͐M����CH�̍ő�l���o�͂���.�e���͂�X,Y,Z�͂��ׂē���ł���K�v������
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeMax_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeMax_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace MergeMax {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �}�[�W���
		  * ID   : MergeType
		  * Text : �}�[�W����ۂ�CH�����ǂ̂悤�Ɍ��肷�邩
		  */
		enum : S32{
			/** Name : �ő�
			  * ID   : max
			  * Text : ���̓��C���[�̍ő吔�ɕ�����
			  */
			MergeType_max,

			/** Name : �ŏ�
			  * ID   : min
			  * Text : ���̓��C���[�̍ŏ����ɕ�����
			  */
			MergeType_min,

			/** Name : �擪���C���[
			  * ID   : layer0
			  * Text : �擪���C���[�̐��ɕ�����
			  */
			MergeType_layer0,

		}MergeType;

	};

} // MergeMax
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_MergeMax_H__
