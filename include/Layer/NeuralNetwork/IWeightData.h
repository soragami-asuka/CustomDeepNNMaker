//========================================
// �p�����[�^���������[�`��
//========================================
#ifndef __GRAVISBELL_I_NN_WEIGHTDATA_H__
#define __GRAVISBELL_I_NN_WEIGHTDATA_H__

#include"../../Common/Common.h"
#include"../../Common/ErrorCode.h"
#include"../../Common/Guiddef.h"
#include"../../Common/IODataStruct.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** ���������[�`�� */
	class IWeightData
	{
	public:
		//===========================
		// �R���X�g���N�^/�f�X�g���N�^
		//===========================
		/** �R���X�g���N�^ */
		IWeightData(){}
		/** �f�X�g���N�^ */
		virtual ~IWeightData(){}

	public:
		//===========================
		// ������
		//===========================
		virtual ErrorCode Initialize(const wchar_t i_initializerID[], U32 i_inputCount, U32 i_outputCount) = 0;

		virtual S64 InitializeFromBuffer(const BYTE* i_lpBuffer, U64 i_bufferSize) = 0;


		//===========================
		// �T�C�Y���擾
		//===========================
		/** Weight�̃T�C�Y���擾���� */
		virtual U64 GetWeigthSize()const = 0;
		/** Bias�̃T�C�Y���擾���� */
		virtual U64 GetBiasSize()const = 0;


		//===========================
		// �l���擾
		//===========================
		/** Weight���擾���� */
		virtual const F32* GetWeight()const = 0;
		/** Bias���擾���� */
		virtual const F32* GetBias()const = 0;


		//===========================
		// �l���X�V
		//===========================
		/** Weigth,Bias��ݒ肷��.
			@param	lpWeight	�ݒ肷��Weight�̒l.
			@param	lpBias		�ݒ肷��Bias�̒l. */
		virtual ErrorCode SetData(const F32* i_lpWeight, const F32* i_lpBias) = 0;
		/** Weight,Bias���X�V����.
			@param	lpDWeight	Weight�̕ω���.
			@param	lpDBias		Bias��h�ω���. */
		virtual ErrorCode UpdateData(const F32* i_lpDWeight, const F32* i_lpDBias) = 0;


		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
		/** �I�v�e�B�}�C�U�[��ύX���� */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]) = 0;


		//===========================
		// ���C���[�ۑ�
		//===========================
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		virtual U64 GetUseBufferByteCount()const = 0;
		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		virtual S64 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell



#endif	__GRAVISBELL_I_NN_INITIALIZER_H__
