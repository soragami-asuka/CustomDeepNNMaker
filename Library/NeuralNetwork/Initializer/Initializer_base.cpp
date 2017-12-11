//=====================================
// �p�����[�^�������N���X.
// ��{�\��. ���̏ꍇ�͂����h�����Ă����Ɗy�ɍ���
//=====================================
#include"stdafx.h"

#include"Initializer_base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;



/** �R���X�g���N�^ */
Initializer_base::Initializer_base()
{
}
/** �f�X�g���N�^1 */
Initializer_base::~Initializer_base()
{
}


//===========================
// �p�����[�^�̒l���擾
//===========================

/** �p�����[�^�̒l���擾����.
	@param	i_inputStruct	���͍\��.
	@param	i_outputStruct	�o�͍\��. */
F32 Initializer_base::GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct)
{
	return this->GetParameter(i_inputStruct.GetDataCount(), i_outputStruct.GetDataCount());
}
/** �p�����[�^�̒l���擾����.
	@param	i_inputStruct	���͍\��.
	@param	i_inputCH		���̓`�����l����.
	@param	i_outputStruct	�o�͍\��.
	@param	i_outputCH		�o�̓`�����l����. */
F32 Initializer_base::GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH)
{
	return this->GetParameter(
		IODataStruct(i_inputCH,  i_inputStruct.x,  i_inputStruct.y,  i_inputStruct.z),
		IODataStruct(i_outputCH, i_outputStruct.x, i_outputStruct.y, i_outputStruct.z) );
}

/** �p�����[�^�̒l���擾����.
	@param	i_inputCount		���͐M����.
	@param	i_outputCount		�o�͐M����.
	@param	i_parameterCount	�ݒ肷��p�����[�^��.
	@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
ErrorCode Initializer_base::GetParameter(U32 i_inputCount, U32 i_outputCount, U32 i_parameterCount, F32 o_lpParameter[])
{
	for(U32 i=0; i<i_parameterCount; i++)
	{
		o_lpParameter[i] = this->GetParameter(i_inputCount, i_outputCount);
	}

	return ErrorCode::ERROR_CODE_NONE;
}
/** �p�����[�^�̒l���擾����.
	@param	i_inputStruct		���͍\��.
	@param	i_outputStruct		�o�͍\��.
	@param	i_parameterCount	�ݒ肷��p�����[�^��.
	@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
ErrorCode Initializer_base::GetParameter(const IODataStruct& i_inputStruct, const IODataStruct& i_outputStruct, U32 i_parameterCount, F32 o_lpParameter[])
{
	return this->GetParameter(i_inputStruct.GetDataCount(), i_outputStruct.GetDataCount(), i_parameterCount, o_lpParameter);
}
/** �p�����[�^�̒l���擾����.
	@param	i_inputStruct		���͍\��.
	@param	i_inputCH			���̓`�����l����.
	@param	i_outputStruct		�o�͍\��.
	@param	i_outputCH			�o�̓`�����l����.
	@param	i_parameterCount	�ݒ肷��p�����[�^��.
	@param	o_lpParameter		�ݒ肷��p�����[�^�̔z��. */
ErrorCode Initializer_base::GetParameter(const Vector3D<S32>& i_inputStruct, U32 i_inputCH, const Vector3D<S32>& i_outputStruct, U32 i_outputCH, U32 i_parameterCount, F32 o_lpParameter[])
{
	return this->GetParameter(
		IODataStruct(i_inputCH,  i_inputStruct.x,  i_inputStruct.y,  i_inputStruct.z),
		IODataStruct(i_outputCH, i_outputStruct.x, i_outputStruct.y, i_outputStruct.z),
		i_parameterCount, o_lpParameter);
}