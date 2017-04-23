/*--------------------------------------------
 * FileName  : Convolution_DATA.hpp
 * LayerName : ��݂��݃j���[�����l�b�g���[�N
 * guid      : F6662E0E-1CA4-4D59-ACCA-CAC29A16C0AA
 * 
 * Text      : ��݂��݃j���[�����l�b�g���[�N.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Convolution_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Convolution_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Convolution {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �o�̓`�����l����
		  * ID   : Output_Channel
		  * Text : �o�͂����`�����l���̐�
		  */
		S32 Output_Channel;

		/** Name : �h���b�v�A�E�g��
		  * ID   : DropOut
		  * Text : �O���C���[�𖳎����銄��.
		  *       : 1.0�őO���C���[�̑S�o�͂𖳎�����
		  */
		F32 DropOut;

		/** Name : �t�B���^�T�C�Y
		  * ID   : FilterSize
		  * Text : ��݂��݂��s�����͐M����
		  */
		Vector3D<S32> FilterSize;

		/** Name : �t�B���^�ړ���
		  * ID   : Move
		  * Text : 1�j���[�������ƂɈړ�������͐M���̈ړ���
		  */
		Vector3D<F32> Move;

		/** Name : ��݂��݈ړ���
		  * ID   : Stride
		  * Text : ��݂��݂��ƂɈړ�������͐M���̈ړ���
		  */
		Vector3D<F32> Stride;

		/** Name : �p�f�B���O�T�C�Y(-����)
		  * ID   : PaddingM
		  */
		Vector3D<S32> PaddingM;

		/** Name : �p�f�B���O�T�C�Y(+����)
		  * ID   : PaddingP
		  */
		Vector3D<S32> PaddingP;

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

			/** Name : ���E�l
			  * ID   : border
			  * Text : �s�����Ɨאڂ���l���Q�Ƃ���
			  */
			PaddingType_border,

			/** Name : ���]
			  * ID   : mirror
			  * Text : �s�����Ɨאڂ���l����t�����ɎQ�Ƃ���
			  */
			PaddingType_mirror,

			/** Name : �N�����v
			  * ID   : clamp
			  * Text : �s�����̔��Α��̋��ڂ��珇�����ɎQ�Ƃ���
			  */
			PaddingType_clamp,

		}PaddingType;

	};

	/** Learning data structure */
	struct LearnDataStructure
	{
		/** Name : �w�K�W��
		  * ID   : LearnCoeff
		  */
		F32 LearnCoeff;

	};

} // Convolution
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Convolution_H__
