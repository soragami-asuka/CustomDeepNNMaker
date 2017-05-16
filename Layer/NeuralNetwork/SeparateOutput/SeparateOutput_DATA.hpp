/*--------------------------------------------
 * FileName  : SeparateOutput_DATA.hpp
 * LayerName : �o�͐M���������C���[
 * guid      : C13C30DA-056E-46D0-90FC-608766FB432E
 * 
 * Text      : �o�͐M���𕪊����郌�C���[.
 *           : �e�������ł̏o�͐M�����ɕω��͂Ȃ����A�덷�̃}�[�W���s�����Ƃ��ł���.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SeparateOutput_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SeparateOutput_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace SeparateOutput {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �o�͐M���̕�����
		  * ID   : separateCount
		  * Text : �o�͐M�������ɕ������邩
		  */
		S32 separateCount;

	};

} // SeparateOutput
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SeparateOutput_H__
