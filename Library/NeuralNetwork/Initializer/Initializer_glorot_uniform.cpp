//=====================================
// �p�����[�^�������N���X.
// Glorot�̈�l���z. limit  = sqrt(6 / (fan_in + fan_out))�̈�l���z
//=====================================
#include"stdafx.h"

#include"Initializer_glorot_uniform.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_glorot_uniform::Initializer_glorot_uniform(Random& random)
	:	random	(random)
{
}
/** �f�X�g���N�^1 */
Initializer_glorot_uniform::~Initializer_glorot_uniform()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_glorot_uniform::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	F32 limit = sqrtf(6.0f / (i_inputCount + i_outputCount));

	return random.GetUniformValue(-limit, +limit);
}

