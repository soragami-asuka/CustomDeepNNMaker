//=====================================
// �p�����[�^�������N���X.
// average=0.0 variance=1.0�̐��K�����ŏ�����
//=====================================
#include"stdafx.h"

#include"Initializer_normal.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_normal::Initializer_normal(Random& random)
	:	random	(random)
{
}
/** �f�X�g���N�^1 */
Initializer_normal::~Initializer_normal()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_normal::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return random.GetNormalValue(0.0f, 1.0f);
}

