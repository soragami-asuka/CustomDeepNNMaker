//=======================================
// �j���[�����l�b�g���[�N�{�̒�`
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNMult2SingleLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INeuralNetwork : public INNMult2SingleLayer
	{
	public:
		/** �R���X�g���N�^ */
		INeuralNetwork(){}
		/** �f�X�g���N�^ */
		virtual ~INeuralNetwork(){}

	public:
		//====================================
		// �w�K�ݒ�
		//====================================
		/** �w�K�ݒ���擾����.
			@param	guid	�擾�Ώۃ��C���[��GUID. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter(const Gravisbell::GUID& guid)const = 0;

		/** �w�K�ݒ��ݒ肷��.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID.	�w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, S32 i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			int�^�Afloat�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID. �w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, F32 i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			bool�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID. �w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, bool i_param) = 0;
		/** �w�K�ݒ��ݒ肷��.
			string�^���Ώ�.
			@param	guid		�擾�Ώۃ��C���[��GUID. �w�肪�����ꍇ�͑S�Ẵ��C���[�ɑ΂��Ď��s����.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;
		virtual ErrorCode SetRuntimeParameter(const Gravisbell::GUID& guid, const wchar_t* i_dataID, const wchar_t* i_param) = 0;

	public:
		//==========================================
		// ���Z����.
		// ���o�͂�CPU���̃������[
		//==========================================
		/** ���Z���������s����.
			@param i_lppInputBuffer		���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ErrorCode Calculate(CONST_BATCH_BUFFER_POINTER i_lppInputBuffer[]) = 0;

		/** ���͌덷�v�Z�������s����.�w�K�����ɓ��͌덷���擾�������ꍇ�Ɏg�p����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	o_lppDInputBuffer	���͌덷�����i�[�惌�C���[.	[GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode CalculateDInput(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

		/** �w�K���������s����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ErrorCode Training(BATCH_BUFFER_POINTER o_lppDInputBuffer[], CONST_BATCH_BUFFER_POINTER i_lppDOutputBuffer) = 0;

	public:
		//==========================================
		// �o�̓o�b�t�@�̎擾
		//==========================================		
		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual CONST_BATCH_BUFFER_POINTER GetOutputBuffer()const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param o_lpOutputBuffer	�o�̓f�[�^�i�[��z��. [GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0 */
		virtual ErrorCode GetOutputBuffer(BATCH_BUFFER_POINTER o_lpOutputBuffer)const = 0;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
