//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __GRAVISBELL_I_OUTPUT_LAYER_H__
#define __GRAVISBELL_I_OUTPUT_LAYER_H__

#include"../../Common/ErrorCode.h"
#include"../../Common/IODataStruct.h"

#include"../ILayerBase.h"

namespace Gravisbell {
namespace Layer {
namespace IO {

	/** ���C���[�x�[�X */
	class IOutputLayer : public virtual ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		IOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IOutputLayer(){}
	};

}	// IO
}	// Layer
}	// GravisBell

#endif