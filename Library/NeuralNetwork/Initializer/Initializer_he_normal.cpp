//=====================================
// �p�����[�^�������N���X.
// He�̐��K���z. stddev = sqrt(2 / fan_in)�̐ؒf���K���z
//=====================================
#include"stdafx.h"

#include"Initializer_he_normal.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_he_normal::Initializer_he_normal(Random& random)
	:	random	(random)
{
}
/** �f�X�g���N�^1 */
Initializer_he_normal::~Initializer_he_normal()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_he_normal::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetTruncatedNormalValue(0.0f, sqrtf(2.0f / i_inputCount) );
}

