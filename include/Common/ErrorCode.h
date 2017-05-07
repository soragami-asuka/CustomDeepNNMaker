//=======================================
// �G���[�R�[�h
//=======================================
#ifndef __GRAVISBELL_ERROR_CODE_H__
#define __GRAVISBELL_ERROR_CODE_H__

#include"Common.h"

namespace Gravisbell {

	enum ErrorCode : U32
	{
		ERROR_CODE_NONE = 0,

		// ���ʌn�G���[
		ERROR_CODE_COMMON = 0x01000000,
		ERROR_CODE_COMMON_NULL_REFERENCE,			///< NULL�Q��
		ERROR_CODE_COMMON_OUT_OF_ARRAYRANGE,		///< �z��O�Q��
		ERROR_CODE_COMMON_OUT_OF_VALUERANGE,		///< �l�͈͂̊O
		ERROR_CODE_COMMON_ALLOCATION_MEMORY,		///< �������̊m�ۂɎ��s
		ERROR_CODE_COMMON_FILE_NOT_FOUND,			///< �t�@�C���̎Q�ƂɎ��s
		ERROR_CODE_COMMON_CALCULATE_NAN,			///< ���Z����NAN����������
		ERROR_CODE_COMMON_NOT_EXIST,				///< ���݂��Ȃ�
		ERROR_CODE_COMMON_ADD_ALREADY_SAMEID,		///< ���ɓ���ID���o�^�ς�
		ERROR_CODE_COMMON_NOT_COMPATIBLE,			///< ���Ή�

		// DLL�n�G���[
		ERROR_CODE_DLL = 0x02000000,
		ERROR_CODE_DLL_LOAD_FUNCTION,			///< �֐��̓ǂݍ��݂Ɏ��s
		ERROR_CODE_DLL_ADD_ALREADY_SAMEID,		///< ���ɓ���ID��DLL���o�^�ς�
		ERROR_CODE_DLL_ERASE_NOTFOUND,			///< �Ώۂ�ID�����Ɏ��s

		// ���o�͌n�G���[
		ERROR_CODE_IO = 0x03000000,
		ERROR_CODE_IO_DISAGREE_INPUT_OUTPUT_COUNT,	///< ���o�͂̃f�[�^������v���Ȃ�

		// CUDA�n�G���[
		ERROR_CODE_CUDA = 0x04000000,
		ERROR_CODE_CUDA_INITIALIZE,				///< CUDA�̏������Ɏ��s
		ERROR_CODE_CUDA_ALLOCATION_MEMORY,		///< �������̊m�ۂɎ��s
		ERROR_CODE_CUDA_COPY_MEMORY,			///< �������̃R�s�[�Ɏ��s
		ERROR_CODE_CUDA_CALCULATE,				///< ���Z���s


		// ���C���[�n�G���[
		ERROR_CODE_LAYER = 0x10000000,
		ERROR_CODE_NONREGIST_CONFIG,			///< �ݒ��񂪓o�^����Ă��Ȃ�
		ERROR_CODE_FRAUD_INPUT_COUNT,			///< ���͐����s��
		ERROR_CODE_FRAUD_OUTPUT_COUNT,			///< �o�͐����s��
		ERROR_CODE_FRAUD_NEURON_COUNT,			///< �j���[���������s��
		// ���C���[�ǉ��n�G���[
		ERROR_CODE_ADDLAYER = 0x10010000,
		ERROR_CODE_ADDLAYER_ALREADY_SAMEID,		///< ����ID�̃��C���[�����ɓo�^�ς�
		ERROR_CODE_ADDLAYER_UPPER_LIMIT,		///< ���C���[�̒ǉ�����ɒB���Ă���
		ERROR_CODE_ADDLAYER_NOT_COMPATIBLE,		///< ���Ή�
		ERROR_CODE_ADDLAYER_NOT_EXIST,			///< ���C���[�����݂��Ȃ�
		// ���C���[�폜�n�G���[
		ERROR_CODE_ERASELAYER = 0x10020000,
		ERROR_CODE_ERASELAYER_NOTFOUND,			///< �Ώۂ�ID�����Ɏ��s
		// ���C���[�������n�G���[
		ERROR_CODE_INITLAYER_DISAGREE_CONFIG,	///< �ݒ���̌^���s��v
		ERROR_CODE_INITLAYER_READ_CONFIG,		///< �ݒ���̓ǂݎ��Ɏ��s


		// ���[�U�[��`
		ERROR_CODE_USER = 0x80000000
	};

}	// Gravisbell

#endif