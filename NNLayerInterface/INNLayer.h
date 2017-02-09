//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_NN_LAYER_BASE_H__
#define __I_NN_LAYER_BASE_H__

#include"IInputLayer.h"
#include"IOutputLayer.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class INNLayer : public IInputLayer, public virtual IOutputLayer
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
			@return	���������ꍇ0 */
		virtual ELayerErrorCode Initialize(const INNLayerConfig& config) = 0;

		/** �ݒ����ݒ� */
		virtual ELayerErrorCode SetLayerConfig(const INNLayerConfig& config) = 0;
		/** ���C���[�̐ݒ�����擾���� */
		virtual const INNLayerConfig* GetLayerConfig()const = 0;

		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		virtual unsigned int GetUseBufferByteCount()const = 0;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ0 */
		virtual int WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		/** ���C���[��ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;

	public:
		/** ���Z���������s����.
			@param lpInputBuffer	���̓f�[�^�o�b�t�@. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0���Ԃ� */
		virtual ELayerErrorCode Calculate() = 0;

		/** �w�K���������C���[�ɔ��f������ */
		virtual ELayerErrorCode ReflectionLearnError(void) = 0;
	};
}

#endif