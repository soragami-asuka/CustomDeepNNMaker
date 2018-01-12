/*--------------------------------------------
 * FileName  : MergeAverage_DATA.hpp
 * LayerName : ���C���[�̃}�[�W(����)
 * guid      : 4E993B4B-9F7A-4CEF-A4C4-37B916BFD9B2
 * 
 * Text      : ���͐M����CH�𕽋ς��ďo�͂���.�e���͂�X,Y,Z�͂��ׂē���ł���K�v������.ch���s�����镔����0����͂��ꂽ���̂Ƃ��Ĉ���
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeAverage_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_MergeAverage_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace MergeAverage {

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

} // MergeAverage
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_MergeAverage_H__
