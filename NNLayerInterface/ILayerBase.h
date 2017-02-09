//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __I_LAYER_BASE_H__
#define __I_LAYER_BASE_H__

#include<guiddef.h>

#include"LayerErrorCode.h"
#include"IODataStruct.h"
#include"INNLayerConfig.h"

#ifndef BYTE
typedef unsigned char BYTE;
#endif

namespace CustomDeepNNLibrary
{
	/** ���C���[��� */
	enum ELayerKind
	{
		LAYER_KIND_CPU = 0x00 << 8,	/**< CPU�������C���[ */
		LAYER_KIND_GPU = 0x01 << 8,	/**< GPU�������C���[ */

		LAYER_KIND_INPUT  = 0x00 << 0,	/**< ���̓��C���[ */
		LAYER_KIND_OUTPUT = 0x01 << 0,	/**< �o�̓��C���[ */
		LAYER_KIND_CALC   = 0x02 << 0,	/**< �v�Z���C���[,���ԑw */

		LAYER_KIND_CPU_INPUT  = LAYER_KIND_CPU | LAYER_KIND_INPUT,
		LAYER_KIND_CPU_OUTPUT = LAYER_KIND_CPU | LAYER_KIND_OUTPUT,
		LAYER_KIND_CPU_CALC   = LAYER_KIND_CPU | LAYER_KIND_CALC,
		
		LAYER_KIND_GPU_INPUT  = LAYER_KIND_GPU | LAYER_KIND_INPUT,
		LAYER_KIND_GPU_OUTPUT = LAYER_KIND_GPU | LAYER_KIND_OUTPUT,
		LAYER_KIND_GPU_CALC   = LAYER_KIND_GPU | LAYER_KIND_CALC,
	};

	/** ���C���[�x�[�X */
	class ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerBase(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerBase(){}

	public:
		/** ���C���[��ʂ̎擾 */
		virtual ELayerKind GetLayerKind()const = 0;

		/** ���C���[�ŗL��GUID���擾���� */
		virtual ELayerErrorCode GetGUID(GUID& o_guid)const = 0;
		
		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_layerCode)const = 0;

	public:
		/** ���Z�O���������s����.
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ELayerErrorCode PreCalculate(unsigned int batchSize) = 0;

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		virtual unsigned int GetBatchSize()const = 0;
	};
}

#endif