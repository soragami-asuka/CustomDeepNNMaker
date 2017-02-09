//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_OUTPUT_LAYER_H__
#define __I_OUTPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class IOutputLayer : public virtual ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		IOutputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IOutputLayer(){}

	public:
		/** �o�̓f�[�^�\�����擾���� */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** �o�̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual unsigned int GetOutputBufferCount()const = 0;

		/** �o�̓f�[�^�o�b�t�@���擾����.
			�z��̗v�f����GetOutputBufferCount�̖߂�l.
			@return �o�̓f�[�^�z��̐擪�|�C���^ */
		virtual const float* GetOutputBuffer()const = 0;
		/** �o�̓f�[�^�o�b�t�@���擾����.
			@param lpOutputBuffer	�o�̓f�[�^�i�[��z��. GetOutputBufferCount�Ŏ擾�����l�̗v�f�����K�v
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetOutputBuffer(float lpOutputBuffer[])const = 0;

	public:
		/** �o�͐惌�C���[�ւ̃����N��ǉ�����.
			@param	pLayer	�ǉ�����o�͐惌�C���[
			@return	���������ꍇ0 */
		virtual ELayerErrorCode AddOutputToLayer(class IInputLayer* pLayer) = 0;
		/** �o�͐惌�C���[�ւ̃����N���폜����.
			@param	pLayer	�폜����o�͐惌�C���[
			@return	���������ꍇ0 */
		virtual ELayerErrorCode EraseOutputToLayer(class IInputLayer* pLayer) = 0;

	public:
		/** �o�͐惌�C���[�����擾���� */
		virtual unsigned int GetOutputToLayerCount()const = 0;
		/** �o�͐惌�C���[�̃A�h���X��ԍ��w��Ŏ擾����.
			@param num	�擾���郌�C���[�̔ԍ�.
			@return	���������ꍇ�o�͐惌�C���[�̃A�h���X.���s�����ꍇ��NULL���Ԃ�. */
		virtual class IInputLayer* GetOutputToLayerByNum(unsigned int num)const = 0;
	};
}

#endif