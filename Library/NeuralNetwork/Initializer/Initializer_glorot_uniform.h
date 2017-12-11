//=====================================
// �p�����[�^�������N���X.
// Glorot�̈�l���z. limit  = sqrt(6 / (fan_in + fan_out))�̈�l���z
//=====================================
#ifndef __GRAVISBELL_NN_INITIALIZER_GLOROT_UNIFORM_H__
#define __GRAVISBELL_NN_INITIALIZER_GLOROT_UNIFORM_H__

#include"Initializer_base.h"
#include"RandomUtility.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Initializer_glorot_uniform : public Initializer_base
	{
	private:
		Random& random;

	public:
		/** �R���X�g���N�^ */
		Initializer_glorot_uniform(Random& random);
		/** �f�X�g���N�^1 */
		virtual ~Initializer_glorot_uniform();


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