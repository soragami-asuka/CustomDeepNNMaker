//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_INPUT_LAYER_H__
#define __I_INPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class IInputLayer : public virtual ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		IInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IInputLayer(){}
	};
}

#endif