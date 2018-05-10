/*--------------------------------------------
 * FileName  : Value2SignalArray_DATA.hpp
 * LayerName : �M���̔z�񂩂�l�֕ϊ�
 * guid      : 6F6C75B8-9C41-43EA-8F80-98C6F1CF4A2D
 * 
 * Text      : �M���̔z�񂩂�l�֕ϊ�����.
 *           : �ő�l�����CH�ԍ���l�ɕϊ�����.
 *           : ����CH��������\�̐����{�ł���K�v������.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Value2SignalArray_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Value2SignalArray_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Value2SignalArray {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ���͍ŏ��l
		  * ID   : inputMinValue
		  */
		F32 inputMinValue;

		/** Name : ���͍ő�l
		  * ID   : inputMaxValue
		  */
		F32 inputMaxValue;

		/** Name : ����\
		  * ID   : resolution
		  */
		S32 resolution;

	};

} // Value2SignalArray
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Value2SignalArray_H__
