//=====================================
// �p�����[�^�������N���X.
// -1�`+1�̈�l�����ŏ�����
//=====================================
#include"stdafx.h"

#include"Initializer_uniform.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_uniform::Initializer_uniform(Random& random)
	:	random	(random)
{
}
/** �f�X�g���N�^1 */
Initializer_uniform::~Initializer_uniform()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_uniform::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetUniformValue(-1.0f, +1.0f);
}

