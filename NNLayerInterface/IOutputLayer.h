//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_OUTPUT_LAYER_H__
#define __I_OUTPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class IOutputLayer : public virtual ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		IOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IOutputLayer(){}
	};
}

#endif