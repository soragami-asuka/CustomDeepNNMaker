//=======================================
// ���o�͐M���f�[�^���C���[
//=======================================
#ifndef __I_INPUT_DATA_LAYER_H__
#define __I_INPUT_DATA_LAYER_H__

#include"IOutputLayer.h"
#include"IInputLayer.h"
#include"IDataLayer.h"

namespace CustomDeepNNLibrary
{
	/** ���o�̓f�[�^���C���[ */
	class IIODataLayer : public IOutputLayer, public IInputLayer, public IDataLayer
	{
	public:
		/** �R���X�g���N�^ */
		IIODataLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IIODataLayer(){}

	};
}

#endif