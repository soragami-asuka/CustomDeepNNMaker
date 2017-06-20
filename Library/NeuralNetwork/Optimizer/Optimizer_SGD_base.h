//===============================================
// �œK�����[�`��(SGD)
//===============================================

#include"Layer/NeuralNetwork/IOptimizer.h"

#include<string>

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class Optimizer_SGD_base : public IOptimizer
	{
	public:
		static const std::wstring OPTIMIZER_ID;

	protected:
		U32 m_parameterCount;	/**< �p�����[�^�� */
		F32 m_learnCoeff;	/**< �w�K�W�� */

	public:
		/** �R���X�g���N�^ */
		Optimizer_SGD_base(U32 i_parameterCount);
		/** �f�X�g���N�^ */
		virtual ~Optimizer_SGD_base();

	public:
		//===========================
		// ��{���
		//===========================
		/** ����ID�̎擾 */
		const wchar_t* GetOptimizerID()const;

		/** �n�C�p�[�p�����[�^��ݒ肷��
			@param	i_parameterID	�p�����[�^���ʗpID
			@param	i_value			�p�����[�^. */
		ErrorCode SetHyperParameter(const wchar_t i_parameterID[], F32 i_value);
		/** �n�C�p�[�p�����[�^��ݒ肷��
			@param	i_parameterID	�p�����[�^���ʗpID
			@param	i_value			�p�����[�^. */
		ErrorCode SetHyperParameter(const wchar_t i_parameterID[], S32 i_value);
		/** �n�C�p�[�p�����[�^��ݒ肷��
			@param	i_parameterID	�p�����[�^���ʗpID
			@param	i_value			�p�����[�^. */
		ErrorCode SetHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]);


	public:
		//===========================
		// �ۑ�
		//===========================
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		U32 GetUseBufferByteCount()const;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		S32 WriteToBuffer(BYTE* o_lpBuffer)const;
	};


	/** �I�v�e�B�}�C�U�[���X�V����.�قȂ�^�������ꍇ�͋����I�Ɏw��̌^�ɕϊ������. */
	ErrorCode ChangeOptimizer_SGD_CPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount);
	ErrorCode ChangeOptimizer_SGD_GPU(IOptimizer** io_ppOptimizer, U32 i_parameterCount);

	/** �I�v�e�B�}�C�U���o�b�t�@����쐬���� */
	IOptimizer* CreateOptimizerFromBuffer_SGD(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize, IOptimizer* (*CreateOptimizer_SGD)(U32) );
	IOptimizer* CreateOptimizerFromBuffer_SGD_CPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize);
	IOptimizer* CreateOptimizerFromBuffer_SGD_GPU(const BYTE* i_lpBuffer, Gravisbell::S32 i_bufferSize, Gravisbell::S32& o_useBufferSize);

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
