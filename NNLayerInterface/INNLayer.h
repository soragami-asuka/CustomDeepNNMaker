//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_NN_LAYER_BASE_H__
#define __I_NN_LAYER_BASE_H__

#include"ISingleInputLayer.h"
#include"IOutputLayer.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class INNLayer : public ISingleInputLayer, public virtual IOutputLayer
	{
	public:
		/** �R���X�g���N�^ */
		INNLayer(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayer(){}

	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		virtual ELayerErrorCode Initialize(void) = 0;
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@param	i_config			�ݒ���
			@oaram	i_inputDataStruct	���̓f�[�^�\�����
			@return	���������ꍇ0 */
		virtual ELayerErrorCode Initialize(const INNLayerConfig& i_config, const IODataStruct& i_inputDataStruct) = 0;
		/** ������. �o�b�t�@����f�[�^��ǂݍ���
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���������ꍇ0 */
		virtual ELayerErrorCode InitializeFromBuffer(BYTE* i_lpBuffer, int i_bufferSize) = 0;

		/** ���C���[�̐ݒ�����擾���� */
		virtual const INNLayerConfig* GetLayerConfig()const = 0;

		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;

	public:
		/** ���Z���������s����.
			@param i_lppInputBuffer	���̓f�[�^�o�b�t�@. [GetBatchSize()�̖߂�l][GetInputBufferCount()�̖߂�l]�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ELayerErrorCode Calculate(const float** i_lppInputBuffer) = 0;

		/** �w�K�덷���v�Z����.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			@param	i_lppDOutputBuffer	�o�͌덷����=�����C���[�̓��͌덷����.	[GetBatchSize()�̖߂�l][GetOutputBufferCount()�̖߂�l]�̗v�f�����K�v.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ELayerErrorCode CalculateLearnError(const float** i_lppDOutputBuffer) = 0;

		/** �w�K���������C���[�ɔ��f������.
			���͐M���A�o�͐M���͒��O��Calculate�̒l���Q�Ƃ���.
			�o�͌덷�����A���͌덷�����͒��O��CalculateLearnError�̒l���Q�Ƃ���. */
		virtual ELayerErrorCode ReflectionLearnError(void) = 0;
	};
}

#endif