/*--------------------------------------------
 * FileName  : Pooling_DATA.hpp
 * LayerName : Pooling
 * guid      : EB80E0D0-9D5A-4ED1-A80D-A1667DE0C890
 * 
 * Text      : Pooling.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Pooling_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Pooling_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Pooling {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �t�B���^�T�C�Y
		  * ID   : FilterSize
		  * Text : Pooling���s���͈�
		  */
		Vector3D<S32> FilterSize;

		/** Name : �t�B���^�ړ���
		  * ID   : Stride
		  * Text : ��݂��݂��ƂɈړ�����t�B���^�̈ړ���
		  */
		Vector3D<S32> Stride;

		/** Name : Pooling���
		  * ID   : PoolingType
		  * Text : Pooling�̕��@�ݒ�
		  */
		enum : S32{
			/** Name : MAX�v�[�����O
			  * ID   : max
			  * Text : �͈͓��̍ő�l���g�p����
			  */
			PoolingType_max,

		}PoolingType;

	};

} // Pooling
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Pooling_H__
