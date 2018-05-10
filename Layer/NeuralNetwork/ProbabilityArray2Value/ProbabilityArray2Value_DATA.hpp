/*--------------------------------------------
 * FileName  : ProbabilityArray2Value_DATA.hpp
 * LayerName : �m���̔z�񂩂�l�֕ϊ�
 * guid      : 9E32D735-A29D-4636-A9CE-2C781BA7BE8E
 * 
 * Text      : �m���̔z�񂩂�l�֕ϊ�����.
 *           : �ő�l�����CH�ԍ���l�ɕϊ�����.
 *           : ����CH��������\�̐����{�ł���K�v������.
 *           : �w�K���̓��͂ɑ΂��鋳�t�M���͐���M���𒆐S�Ƃ������K���z�̕��ϒl���Ƃ�
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ProbabilityArray2Value_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_ProbabilityArray2Value_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace ProbabilityArray2Value {

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

		/** Name : ���t�M���̕��U
		  * ID   : variance
		  */
		F32 variance;

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

} // ProbabilityArray2Value
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_ProbabilityArray2Value_H__
