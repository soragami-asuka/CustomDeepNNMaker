/*--------------------------------------------
 * FileName  : LimitBackPropagationRange_DATA.hpp
 * LayerName : ����`������͈͂𐧌�����
 * guid      : 89359466-E1B2-4129-90E8-50C74E4BC597
 * 
 * Text      : ����`������͈͂𐧌�����.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_LimitBackPropagationRange_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_LimitBackPropagationRange_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace LimitBackPropagationRange {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �J�n�ʒu
		  * ID   : startPosition
		  */
		Vector3D<S32> startPosition;

		/** Name : ���oXYZ�T�C�Y
		  * ID   : boxSize
		  */
		Vector3D<S32> boxSize;

	};

} // LimitBackPropagationRange
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_LimitBackPropagationRange_H__
