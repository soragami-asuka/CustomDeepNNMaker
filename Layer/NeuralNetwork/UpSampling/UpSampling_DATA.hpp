/*--------------------------------------------
 * FileName  : UpSampling_DATA.hpp
 * LayerName : �A�b�v�T���v�����O
 * guid      : 14EEE4A7-1B26-4651-8EBF-B1156D62CE1B
 * 
 * Text      : �l���g�����A�����߂���
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpSampling_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_UpSampling_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace UpSampling {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �g����
		  * ID   : UpScale
		  */
		Vector3D<S32> UpScale;

		/** Name : �p�f�B���O���
		  * ID   : PaddingType
		  * Text : �p�f�B���O���s���ۂ̕��@�ݒ�
		  */
		enum : S32{
			/** Name : �[���p�f�B���O
			  * ID   : zero
			  * Text : �s������0�Ŗ��߂�
			  */
			PaddingType_zero,

			/** Name : �l
			  * ID   : value
			  * Text : �s�����Ɨאڂ���l���Q�Ƃ���
			  */
			PaddingType_value,

		}PaddingType;

	};

} // UpSampling
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_UpSampling_H__
