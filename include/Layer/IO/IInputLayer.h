//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __GRAVISBELL_I_INPUT_LAYER_H__
#define __GRAVISBELL_I_INPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** ���C���[�x�[�X */
	class IInputLayer : public virtual ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		IInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IInputLayer(){}
	};

}	// IO
}	// Layer
}	// Gravisbell

#endif