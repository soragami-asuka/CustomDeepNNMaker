//=====================================
// �p�����[�^�������N���X.
// �S��1�ŏ�����
//=====================================
#include"stdafx.h"

#include"Initializer_one.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_one::Initializer_one()
{
}
/** �f�X�g���N�^1 */
Initializer_one::~Initializer_one()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_one::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return 1.0f;
}

