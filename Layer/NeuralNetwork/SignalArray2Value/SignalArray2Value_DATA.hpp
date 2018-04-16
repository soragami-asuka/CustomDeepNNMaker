/*--------------------------------------------
 * FileName  : SignalArray2Value_DATA.hpp
 * LayerName : �M���̔z�񂩂�l�֕ϊ�
 * guid      : 97C8E5D3-AA0E-43AA-96C2-E7E434F104B8
 * 
 * Text      : �M���̔z�񂩂�l�֕ϊ�����.
 *           : �o��CH����1�ɋ����ϊ�.
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

	};

} // SignalArray2Value
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SignalArray2Value_H__
