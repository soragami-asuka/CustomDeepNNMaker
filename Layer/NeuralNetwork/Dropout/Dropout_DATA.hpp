/*--------------------------------------------
 * FileName  : Dropout_DATA.hpp
 * LayerName : �h���b�v�A�E�g
 * guid      : 298243E4-2111-474F-A8F4-35BDC8764588
 * 
 * Text      : �h���b�v�A�E�g.
--------------------------------------------*/
#ifndef __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Dropout_H__
#define __GRAVISBELL_NEURALNETWORK_LAYER_DATA_Dropout_H__

#include<guiddef.h>

#include<Common/ErrorCode.h>
#include<SettingData/Standard/IData.h>
#include<Layer/NeuralNetwork/INNLayer.h>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {
namespace Dropout {

	/** Layer structure */
	struct LayerStructure
	{
		/** Name : �h���b�v�A�E�g��
		  * ID   : Rate
		  * Text : �O���C���[�𖳎����銄��.
		  *       : 1.0�őO���C���[�̑S�o�͂𖳎�����
		  */
		F32 Rate;

	};

} // Dropout
} // NeuralNetwork
} // Layer
} // Gravisbell


#endif // __CUSTOM_DEEP_NN_LAYER_DATA_Dropout_H__
