//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_INPUT_LAYER_H__
#define __I_INPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** ���C���[�x�[�X */
	class IInputLayer : public virtual ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		IInputLayer(){}
		/** �f�X�g���N�^ */
		virtual ~IInputLayer(){}

	public:
		/** �w�K�덷���v�Z����.
			���O�̌v�Z���ʂ��g�p���� */
		virtual ELayerErrorCode CalculateLearnError() = 0;

	public:
		/** ���̓o�b�t�@�����擾����. byte���ł͖����f�[�^�̐��Ȃ̂Œ��� */
		virtual unsigned int GetInputBufferCount()const = 0;

		/** �w�K�������擾����.
			�z��̗v�f����GetInputBufferCount�̖߂�l.
			@return	�덷�����z��̐擪�|�C���^ */
		virtual const float* GetDInputBuffer()const = 0;
		/** �w�K�������擾����.
			@param lpDOutputBuffer	�w�K�������i�[����z��. GetInputBufferCount�Ŏ擾�����l�̗v�f�����K�v */
		virtual ELayerErrorCode GetDInputBuffer(float o_lpDInputBuffer[])const = 0;

	public:
		/** ���͌����C���[�ւ̃����N��ǉ�����.
			@param	pLayer	�ǉ�������͌����C���[
			@return	���������ꍇ0 */
		virtual ELayerErrorCode AddInputFromLayer(class IOutputLayer* pLayer) = 0;
		/** ���͌����C���[�ւ̃����N���폜����.
			@param	pLayer	�폜������͌����C���[
			@return	���������ꍇ0 */
		virtual ELayerErrorCode EraseInputFromLayer(class IOutputLayer* pLayer) = 0;

	public:
		/** ���͌����C���[�����擾���� */
		virtual unsigned int GetInputFromLayerCount()const = 0;
		/** ���͌����C���[�̃A�h���X��ԍ��w��Ŏ擾����.
			@param num	�擾���郌�C���[�̔ԍ�.
			@return	���������ꍇ���͌����C���[�̃A�h���X.���s�����ꍇ��NULL���Ԃ�. */
		virtual class IOutputLayer* GetInputFromLayerByNum(unsigned int num)const = 0;

		/** ���͌����C���[�����̓o�b�t�@�̂ǂ̈ʒu�ɋ��邩��Ԃ�.
			���Ώۓ��̓��C���[�̑O�ɂ����̓��̓o�b�t�@�����݂��邩.
			�@�w�K�����̎g�p�J�n�ʒu�Ƃ��Ă��g�p����.
			@return ���s�����ꍇ���̒l���Ԃ�*/
		virtual int GetInputBufferPositionByLayer(const class IOutputLayer* pLayer) = 0;
	};
}

#endif