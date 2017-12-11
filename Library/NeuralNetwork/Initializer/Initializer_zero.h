//=====================================
// �p�����[�^�������N���X.
// �S��0�ŏ�����
//=====================================
#ifndef __GRAVISBELL_NN_INITIALIZER_ZERO_H__
#define __GRAVISBELL_NN_INITIALIZER_ZERO_H__

#include"Initializer_base.h"
#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Initializer_zero : public Initializer_base
	{
	public:
		/** �R���X�g���N�^ */
		Initializer_zero();
		/** �f�X�g���N�^1 */
		virtual ~Initializer_zero();


	public:
		//===========================
		// �p�����[�^�̒l���擾
		//===========================
		/** �p�����[�^�̒l���擾����.
			@param	i_inputCount	���͐M����.
			@param	i_outputCount	�o�͐M����. */
		F32 GetParameter(U32 i_inputCount, U32 i_outputCount);
	};


}	// NeuralNetwork
}	// Layer
}	// Gravisbell

#endif	__GRAVISBELL_NN_INITIALIZER_ZERO_H__