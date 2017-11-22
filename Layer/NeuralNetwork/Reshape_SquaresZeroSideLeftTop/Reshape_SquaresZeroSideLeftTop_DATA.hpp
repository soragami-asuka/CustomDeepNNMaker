/*--------------------------------------------
 * FileName  : Reshape_SquaresZeroSideLeftTop_DATA.hpp
 * LayerName : 
 * guid      : F6D9C5DA-D583-455B-9254-5AEF3CA9021B
 * 
 * Text      : X���W0�𒆐S�ɓ��͐M���𕽕�������.
 *           : X�~Y+1�̓��͐M����K�v�Ƃ���
 *           : X=0 or Y=0�����f�[�^��X=0�Ƃ���
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_SquaresZeroSideLeftTop_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Reshape_SquaresZeroSideLeftTop_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Reshape_SquaresZeroSideLeftTop {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : X
		  * ID   : x
		  * Text : �o�͍\����X�T�C�Y
		  */
		S32 x;

		/** Name : Y
		  * ID   : y
		  * Text : �o�͍\����Y�T�C�Y
		  */
		S32 y;

	};

} // Reshape_SquaresZeroSideLeftTop
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Reshape_SquaresZeroSideLeftTop_H__
