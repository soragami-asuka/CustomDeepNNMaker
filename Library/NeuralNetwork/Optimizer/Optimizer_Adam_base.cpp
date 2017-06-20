//===============================================
// �œK�����[�`��(Adam)
//===============================================
#include"stdafx.h"

#include"Optimizer_Adam_base.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	const std::wstring Optimizer_Adam_base::OPTIMIZER_ID = L"Adam";

	/** �R���X�g���N�^ */
	Optimizer_Adam_base::Optimizer_Adam_base(U32 i_parameterCount)
		:	m_parameterCount	(i_parameterCount)

		,	m_alpha			(0.001f)		/**< ����. */
		,	m_beta1			(0.9f)			/**< ������. */
		,	m_beta2			(0.999f)		/**< ������. */
		,	m_epsilon		(1e-8)			/**< �⏕�W��. */
	{
	}
	/** �f�X�g���N�^ */
	Optimizer_Adam_base::~Optimizer_Adam_base()
	{
	}


	//===========================
	// ��{���
	//===========================
	/** ����ID�̎擾 */
	const wchar_t* Optimizer_Adam_base::GetOptimizerID()const
	{
		return OPTIMIZER_ID.c_str();
	}

	/** �n�C�p�[�p�����[�^��ݒ肷��
		@param	i_parameterID	�p�����[�^���ʗpID
		@param	i_value			�p�����[�^. */
	ErrorCode Optimizer_Adam_base::SetHyperParameter(const wchar_t i_parameterID[], F32 i_value)
	{
		std::wstring parameter = i_parameterID;
		if(parameter == L"alpha")
		{
			this->m_alpha = i_value;
		}
		else if(parameter == L"beta1")
		{
			this->m_beta1 = i_value;
		}
		else if(parameter == L"beta2")
		{
			this->m_beta2 = i_value;
		}
		else if(parameter == L"epsilon")
		{
			this->m_epsilon = i_value;
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
	ErrorCode Optimizer_Adam_base::SetHyperParameter(const wchar_t i_parameterID[], S32 i_value)
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}
	/** �n�C�p�[�p�����[�^��ݒ肷��
		@param	i_parameterID	�p�����[�^���ʗpID
		@param	i_value			�p�����[�^. */
	ErrorCode Optimizer_Adam_base::SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[])
	{
		return ErrorCode::ERROR_CODE_COMMON_NOT_COMPATIBLE;
	}


	//===========================
	// �ۑ�
	//===========================
	/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
	U32 Optimizer_Adam_base::GetUseBufferByteCount()const
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

		// ��
		useBufferByte += sizeof(this->m_alpha);
		// ��1
		useBufferByte += sizeof(this->m_beta1);
		// ��2
		useBufferByte += sizeof(this->m_beta2);
		// ��
		useBufferByte += sizeof(this->m_epsilon);

		// M
		useBufferByte += sizeof(F32) * this->m_parameterCount;
		// V
		useBufferByte += sizeof(F32) * this->m_parameterCount;

		// ��1^n
		useBufferByte += sizeof(F32);
		// ��2^n
		useBufferByte += sizeof(F32);


		return useBufferByte;
	}

	/** ���C���[���o�b�t�@�ɏ�������.
		@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
		@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
	S32 Optimizer_Adam_base::WriteToBufferBase(BYTE* o_lpBuffer)const
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


		// ��
		memcpy(&o_lpBuffer[writePos], &this->m_alpha, sizeof(this->m_alpha));
		writePos+= sizeof(this->m_alpha);
		// ��1
		memcpy(&o_lpBuffer[writePos], &this->m_beta1, sizeof(this->m_beta1));
		writePos+= sizeof(this->m_beta1);
		// ��2
		memcpy(&o_lpBuffer[writePos], &this->m_beta2, sizeof(this->m_beta2));
		writePos+= sizeof(this->m_beta2);
		// ��
		memcpy(&o_lpBuffer[writePos], &this->m_epsilon, sizeof(this->m_epsilon));
		writePos+= sizeof(this->m_epsilon);


		return writePos;
	}

	/** �o�b�t�@����쐬���� */
	Optimizer_Adam_base* CreateOptimizerFromBuffer_Adam(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize, Optimizer_Adam_base* (*CreateOptimizer_Adam)(U32) )
	{
		o_useBufferSize = -1;
		U32 readBufferPos = 0;

		// �g�p�o�b�t�@��, ID�͓ǂݎ��ς�

		// �p�����[�^��
		U32 parameterCount = 0;
		memcpy(&parameterCount, &i_lpBuffer[readBufferPos], sizeof(parameterCount));
		readBufferPos += sizeof(parameterCount);

		// �쐬
		Optimizer_Adam_base* pOptimizer = CreateOptimizer_Adam(parameterCount);
		if(pOptimizer == NULL)
			return NULL;


		// ��
		F32 alpha = 0.0f;
		memcpy(&alpha, &i_lpBuffer[readBufferPos], sizeof(alpha));
		readBufferPos += sizeof(alpha);
		pOptimizer->SetHyperParameter(L"alpha", alpha);
		// ��1
		F32 beta1 = 0.0f;
		memcpy(&beta1, &i_lpBuffer[readBufferPos], sizeof(beta1));
		readBufferPos += sizeof(beta1);
		pOptimizer->SetHyperParameter(L"beta1", beta1);
		// ��2
		F32 beta2 = 0.0f;
		memcpy(&beta2, &i_lpBuffer[readBufferPos], sizeof(beta2));
		readBufferPos += sizeof(beta2);
		pOptimizer->SetHyperParameter(L"beta2", beta2);
		// ��
		F32 epsilon = 0.0f;
		memcpy(&epsilon, &i_lpBuffer[readBufferPos], sizeof(epsilon));
		readBufferPos += sizeof(epsilon);
		pOptimizer->SetHyperParameter(L"epsilon", epsilon);


		// �g�p�o�b�t�@���ۑ�
		o_useBufferSize = readBufferPos;

		return pOptimizer;
	}

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
