/*--------------------------------------------
 * FileName  : SOM_DATA.hpp
 * LayerName : ���ȑg�D���}�b�v
 * guid      : AF36DF4D-9F50-46FF-A1C1-5311CA761F6A
 * 
 * Text      : ���ȑg�D���}�b�v.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SOM_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_SOM_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace SOM {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : ���̓o�b�t�@��
		  * ID   : InputBufferCount
		  * Text : ���C���[�ɑ΂�����̓o�b�t�@��
		  */
		S32 InputBufferCount;

		/** Name : ������
		  * ID   : DimensionCount
		  * Text : ���������}�b�v�̎�����
		  */
		S32 DimensionCount;

		/** Name : ����\
		  * ID   : ResolutionCount
		  * Text : �������Ƃ̕��𐫔\
		  */
		S32 ResolutionCount;

		/** Name : �������ŏ��l
		  * ID   : InitializeMinValue
		  * Text : �������Ɏg�p����l�̍ŏ��l
		  */
		F32 InitializeMinValue;

		/** Name : �������ő�l
		  * ID   : InitializeMaxValue
		  * Text : �������Ɏg�p����l�̍ő�l
		  */
		F32 InitializeMaxValue;

	};

	/** Runtime Parameter structure */
	struct RuntimeParameterStructure
	{
		/** Name : �w�K�W��
		  * ID   : SOM_L0
		  * Text : �p�����[�^�X�V�̌W��
		  */
		F32 SOM_L0;

		/** Name : ���Ԍ�����
		  * ID   : SOM_ramda
		  * Text : �w�K�񐔂ɉ������w�K���̌�����.�l�������ق����������͒Ⴂ
		  */
		F32 SOM_ramda;

		/** Name : ����������
		  * ID   : SOM_sigma
		  * Text : �X�V�̂�BMU�Ƃ̋����ɉ�����������.�l�������ق����������͒Ⴂ
		  */
		F32 SOM_sigma;

	};

} // SOM
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_SOM_H__
