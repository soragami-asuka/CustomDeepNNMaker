//=======================================
// ���C���[�x�[�X
//=======================================
#ifndef __GRAVISBELL_I_LAYER_BASE_H__
#define __GRAVISBELL_I_LAYER_BASE_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

namespace Gravisbell {
namespace Layer {

	/** ���C���[��� */
	enum ELayerKind : U32
	{
		LAYER_KIND_CPU = 0x00 << 16,	/**< CPU�������C���[ */
		LAYER_KIND_GPU = 0x01 << 16,	/**< GPU�������C���[ */

		LAYER_KIND_SINGLE_INPUT  = 0x01 << 0,	/**< ���̓��C���[ */
		LAYER_KIND_MULT_INPUT    = 0x01 << 1,	/**< ���̓��C���[ */
		LAYER_KIND_SINGLE_OUTPUT = 0x01 << 2,	/**< �o�̓��C���[ */
		LAYER_KIND_MULT_OUTPUT   = 0x01 << 3,	/**< �o�̓��C���[ */

		LAYER_KIND_DATA			 = 0x01 << 8,	/**< �f�[�^���C���[.���o�� */
		LAYER_KIND_NEURALNETWORK = 0x02 << 8,	/**< �j���[�����l�b�g���[�N���C���[ */
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
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		virtual unsigned int GetLayerKind()const = 0;

		/** ���C���[�ŗL��GUID���擾���� */
		virtual ErrorCode GetGUID(Gravisbell::GUID& o_guid)const = 0;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual ErrorCode GetLayerCode(Gravisbell::GUID& o_layerCode)const = 0;

	public:
		/** ���Z�O���������s����.(�w�K�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��PreProcessLearnLoop�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLearn(unsigned int batchSize) = 0;
		/** ���Z�O���������s����.(���Z�p)
			@param batchSize	�����ɉ��Z���s���o�b�`�̃T�C�Y.
			NN�쐬��A���Z���������s����O�Ɉ�x�����K�����s���邱�ƁB�f�[�^���ƂɎ��s����K�v�͂Ȃ�.
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessCalculate(unsigned int batchSize) = 0;


		/** �w�K���[�v�̏���������.�f�[�^�Z�b�g�̊w�K�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLearnLoop(const SettingData::Standard::IData& data) = 0;
		/** ���Z���[�v�̏���������.�f�[�^�Z�b�g�̉��Z�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessCalculateLoop() = 0;


		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		virtual unsigned int GetBatchSize()const = 0;
	};

}	// Layer
}	// Gravisbell

#endif