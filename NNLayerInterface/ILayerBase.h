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
		LAYER_KIND_CPU = 0x00 << 16,	/**< CPU�������C���[ */
		LAYER_KIND_GPU = 0x01 << 16,	/**< GPU�������C���[ */

		LAYER_KIND_SINGLE_INPUT  = 0x01 << 0,	/**< ���̓��C���[ */
		LAYER_KIND_MULT_INPUT    = 0x01 << 1,	/**< ���̓��C���[ */
		LAYER_KIND_SINGLE_OUTPUT = 0x01 << 2,	/**< �o�̓��C���[ */
		LAYER_KIND_MULT_OUTPUT   = 0x01 << 3,	/**< �o�̓��C���[ */

		LAYER_KIND_CALC          = 0x01 << 8,	/**< �v�Z���C���[,���ԑw */
		LAYER_KIND_DATA			 = 0x02 << 8,	/**< �f�[�^���C���[.���o�͑w */
	};

	/** ���C���[�Ԃ̃f�[�^�̂������s���o�b�`�����p2�����z��|�C���^�^.
		[�o�b�`�T�C�Y][�o�b�t�@��] */
	typedef float**				BATCH_BUFFER_POINTER;
	/** ���C���[�Ԃ̃f�[�^�̂������s���o�b�`�����p2�����z��|�C���^�^(�萔).
		[�o�b�`�T�C�Y][�o�b�t�@��] */
	typedef const float*const*	CONST_BATCH_BUFFER_POINTER;


	/** ���C���[�x�[�X */
	class ILayerBase
	{
	public:
		/** �R���X�g���N�^ */
		ILayerBase(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerBase(){}

	public:
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		virtual unsigned int GetLayerKind()const = 0;

		/** ���C���[�ŗL��GUID���擾���� */
		virtual ELayerErrorCode GetGUID(GUID& o_guid)const = 0;
		
		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_layerCode)const = 0;

	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		virtual ELayerErrorCode PreProcessLearn(unsigned int batchSize) = 0;

		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ELayerErrorCode PreProcessCalculate(unsigned int batchSize) = 0;

		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ELayerErrorCode PreProcessLearnLoop(const INNLayerConfig& config) = 0;


		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		virtual unsigned int GetBatchSize()const = 0;
	};
}

#endif