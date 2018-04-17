/*--------------------------------------------
 * FileName  : SignalArray2Value_DATA.hpp
 * LayerName : �M���̔z�񂩂�l�֕ϊ�
 * guid      : 97C8E5D3-AA0E-43AA-96C2-E7E434F104B8
 * 
 * Text      : �M���̔z�񂩂�l�֕ϊ�����.
 *           : �ő�l�����CH�ԍ���l�ɕϊ�����.
 *           : ����CH��������\�̐����{�ł���K�v������.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SignalArray2Value_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SignalArray2Value_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace SignalArray2Value {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �o�͍ŏ��l
		  * ID   : outputMinValue
		  */
		F32 outputMinValue;

		/** Name : �o�͍ő�l
		  * ID   : outputMaxValue
		  */
		F32 outputMaxValue;

		/** Name : ����\
		  * ID   : resolution
		  */
		S32 resolution;

		/** Name : ���蓖�Ď��
		  * ID   : allocationType
		  * Text : CH�ԍ����l�ɕϊ����邽�߂̕ϊ����@
		  */
		enum : S32{
			/** Name : �ő�l
			  * ID   : max
			  * Text : CH���̍ő�l���o�͂���
			  */
			allocationType_max,

			/** Name : ����
			  * ID   : average
			  * Text : CH�ԍ���CH�̒l���|�����킹���l�̕��ϒl���o�͂���(��������)
			  */
			allocationType_average,

		}allocationType;

	};

} // SignalArray2Value
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SignalArray2Value_H__
