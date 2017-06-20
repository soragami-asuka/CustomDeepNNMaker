//===============================================
// �œK�����[�`��(SGD)
//===============================================
#include"stdafx.h"

#include"Optimizer_SGD_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	const std::wstring Optimizer_SGD_base::OPTIMIZER_ID = L"SGD";

	/** �R���X�g���N�^ */
	Optimizer_SGD_base::Optimizer_SGD_base(U32 i_parameterCount)
		:	m_parameterCount	(i_parameterCount)
		,	m_learnCoeff		(1.0f)
	{
	}
	/** �f�X�g���N�^ */
	Optimizer_SGD_base::~Optimizer_SGD_base()
	{
	}


	//===========================
	// ��{���
	//===========================
	/** ����ID�̎擾 */
	const wchar_t* Optimizer_SGD_base::GetOptimizerID()const
	{
		return OPTIMIZER_ID.c_str();
	}

	/** �n�C�p�[�p�����[�^��ݒ肷��
		@param	i_parameterID	�p�����[�^���ʗpID
		@param	i_value			�p�����[�^. */
	ErrorCode Optimizer_SGD_base::SetHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		std::wstring parameter = i_parameterID;
		if(parameter == L"LearnCoeff")
		{
			this->m_learnCoeff = i_value;
		}
		else
		{
			return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
		}

		return ErrorCode::ERROR_CODE_NONE;
	}
	/** �n�C�p�[�p�����[�^��ݒ肷��
		@param	i_parameterID	�p�����[�^���ʗpID
		@param	i_value			�p�����[�^. */
	ErrorCode Optimizer_SGD_base::SetHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** �n�C�p�[�p�����[�^��ݒ肷��
		@param	i_parameterID	�p�����[�^���ʗpID
		@param	i_value			�p�����[�^. */
	ErrorCode Optimizer_SGD_base::SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


	//===========================
	// �ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 Optimizer_SGD_base::GetUseBufferByteCount()const
	{
		U32 useBufferByte = 0;

		// �g�p�o�C�g���i�[
		useBufferByte += sizeof(U32);

		// ID�o�b�t�@�T�C�Y
		useBufferByte += sizeof(U32);

		// ID�o�b�t�@
		useBufferByte += sizeof(wchar_t) * OPTIMIZER_ID.size();

		// �p�����[�^��
		useBufferByte += sizeof(this->m_parameterCount);

		// �w�K�W��
		useBufferByte += sizeof(this->m_learnCoeff);

		return useBufferByte;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Optimizer_SGD_base::WriteToBuffer(BYTE* o_lpBuffer)const
	{
		U32 writePos = 0;

		// �g�p�o�C�g��
		U32 userBufferByte = this->GetUseBufferByteCount();
		memcpy(&o_lpBuffer[writePos], &userBufferByte, sizeof(userBufferByte));
		writePos += sizeof(userBufferByte);

		// ID�o�b�t�@�T�C�Y
		U32 idBufferSize = sizeof(wchar_t) * OPTIMIZER_ID.size();
		memcpy(&o_lpBuffer[writePos], &idBufferSize, sizeof(idBufferSize));
		writePos += sizeof(idBufferSize);

		// ID�o�b�t�@
		memcpy(&o_lpBuffer[writePos], (const BYTE*)OPTIMIZER_ID.c_str(), idBufferSize);
		writePos += idBufferSize;

		// �p�����[�^��
		memcpy(&o_lpBuffer[writePos], &this->m_parameterCount, sizeof(this->m_parameterCount));
		writePos+= sizeof(this->m_parameterCount);

		// �w�K�W��
		memcpy(&o_lpBuffer[writePos], &this->m_learnCoeff, sizeof(this->m_learnCoeff));
		writePos+= sizeof(this->m_learnCoeff);

		return writePos;
	}

	/** �o�b�t�@����쐬���� */
	IOptimizer* CreateOptimizerFromBuffer_SGD(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize, IOptimizer* (*CreateOptimizer_SGD)(U32) )
	{
		o_useBufferSize = -1;
		U32 readBufferPos = 0;

		// �g�p�o�b�t�@��, ID�͓ǂݎ��ς�

		// �p�����[�^��
		U32 parameterCount = 0;
		memcpy(&parameterCount, &i_lpBuffer[readBufferPos], sizeof(parameterCount));
		readBufferPos += sizeof(parameterCount);

		// �쐬
		IOptimizer* pOptimizer = CreateOptimizer_SGD(parameterCount);
		if(pOptimizer == NULL)
			return NULL;

		// �w�K�W��
		F32 learnCoeff = 0.0f;
		memcpy(&learnCoeff, &i_lpBuffer[readBufferPos], sizeof(learnCoeff));
		readBufferPos += sizeof(learnCoeff);
		pOptimizer->SetHyperParameter(L"LearnCoeff", learnCoeff);

		// �g�p�o�b�t�@���ۑ�
		o_useBufferSize = readBufferPos;

		return pOptimizer;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
