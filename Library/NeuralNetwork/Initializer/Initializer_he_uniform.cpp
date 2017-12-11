//=====================================
// �p�����[�^�������N���X.
// He�̈�l���z. limit  = sqrt(6 / fan_in)�̈�l���z
//=====================================
#include"stdafx.h"

#include"Initializer_he_uniform.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_he_uniform::Initializer_he_uniform(Random& random)
	:	random	(random)
{
}
/** �f�X�g���N�^1 */
Initializer_he_uniform::~Initializer_he_uniform()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_he_uniform::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	F32 limit = sqrtf(6.0f / i_outputCount);

	return random.GetUniformValue(-limit, +limit);
}

