//=====================================
// �p�����[�^�������N���X.
// �S��0�ŏ�����
//=====================================
#include"stdafx.h"

#include"Initializer_zero.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_zero::Initializer_zero()
{
}
/** �f�X�g���N�^1 */
Initializer_zero::~Initializer_zero()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================
/** �p�����[�^�̒l���擾����.
	@param	i_inputCount	���͐M����.
	@param	i_outputCount	�o�͐M����. */
F32 Initializer_zero::GetParameter(U32 i_inputCount, U32 i_outputCount)
{
	return 0.0f;
}

