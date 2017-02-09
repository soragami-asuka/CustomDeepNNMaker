//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __LAYER_ERROR_CODE_H__
#define __LAYER_ERROR_CODE_H__

namespace CustomDeepNNLibrary
{
	enum ELayerErrorCode
	{
		LAYER_ERROR_NONE = 0,

		// ���ʌn�G���[
		LAYER_ERROR_COMMON = 0x01000000,
		LAYER_ERROR_COMMON_NULL_REFERENCE,		///< NULL�Q��
		LAYER_ERROR_COMMON_OUT_OF_ARRAYRANGE,	///< �z��O�Q��
		LAYER_ERROR_COMMON_OUT_OF_VALUERANGE,	///< �l�͈͂̊O
		LAYER_ERROR_COMMON_ALLOCATION_MEMORY,	///< �������̊m�ۂɎ��s
		LAYER_ERROR_COMMON_FILE_NOT_FOUND,		///< �t�@�C���̎Q�ƂɎ��s

		// ���C���[�n�G���[
		LAYER_ERROR_LAYER = 0x02000000,
		LAYER_ERROR_NONREGIST_CONFIG,			///< �ݒ��񂪓o�^����Ă��Ȃ�
		LAYER_ERROR_FRAUD_INPUT_COUNT,			///< ���͐����s��
		LAYER_ERROR_FRAUD_OUTPUT_COUNT,			///< �o�͐����s��
		LAYER_ERROR_FRAUD_NEURON_COUNT,			///< �j���[���������s��
		// ���C���[�ǉ��n�G���[
		LAYER_ERROR_ADDLAYER = 0x02010000,
		LAYER_ERROR_ADDLAYER_ALREADY_SAMEID,	///< ����ID�̃��C���[�����ɓo�^�ς�
		// ���C���[�폜�n�G���[
		LAYER_ERROR_ERASELAYER = 0x02020000,
		LAYER_ERROR_ERASELAYER_NOTFOUND,		///< �Ώۂ�ID�����Ɏ��s
		// ���C���[�������n�G���[
		LAYER_ERROR_INITLAYER_DISAGREE_CONFIG,	///< �ݒ���̌^���s��v
		LAYER_ERROR_INITLAYER_READ_CONFIG,		///< �ݒ���̓ǂݎ��Ɏ��s

		// DLL�n�G���[
		LAYER_ERROR_DLL = 0x03000000,
		LAYER_ERROR_DLL_LOAD_FUNCTION,		///< �֐��̓ǂݍ��݂Ɏ��s
		LAYER_ERROR_DLL_ADD_ALREADY_SAMEID,	///< ���ɓ���ID��DLL���o�^�ς�
		LAYER_ERROR_DLL_ERASE_NOTFOUND,		///< �Ώۂ�ID�����Ɏ��s

		// ���o�͌n�G���[
		LAYER_ERROR_IO = 0x04000000,
		LAYER_ERROR_IO_DISAGREE_INPUT_OUTPUT_COUNT,	///< ���o�͂̃f�[�^������v���Ȃ�


		// ���[�U�[��`
		LAYER_ERROR_USER = 0x70000000
	};
}

#endif