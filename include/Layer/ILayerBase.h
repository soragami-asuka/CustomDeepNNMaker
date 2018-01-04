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
		LAYER_KIND_IOTYPE		 = 0xFF << 0,
		LAYER_KIND_INPUTTYPE	 = 0x01 << 0,
		LAYER_KIND_OUTPUTTYPE	 = 0x01 << 1,
		LAYER_KIND_SINGLE_INPUT  = (0x00 << 0) << 0,	/**< ���̓��C���[ */
		LAYER_KIND_MULT_INPUT    = (0x01 << 0) << 0,	/**< ���̓��C���[ */
		LAYER_KIND_SINGLE_OUTPUT = (0x00 << 1) << 0,	/**< �o�̓��C���[ */
		LAYER_KIND_MULT_OUTPUT   = (0x01 << 1) << 0,	/**< �o�̓��C���[ */

		LAYER_KIND_USETYPE		 = 0xFF << 8,
		LAYER_KIND_DATA			 = 0x01 << 8,	/**< �f�[�^���C���[.���o�� */
		LAYER_KIND_NEURALNETWORK = 0x02 << 8,	/**< �j���[�����l�b�g���[�N���C���[ */
		
		LAYER_KIND_CALCTYPE = 0x0F << 16,
		LAYER_KIND_UNKNOWN	= 0x00 << 16,
		LAYER_KIND_CPU		= 0x01 << 16,	/**< CPU�������C���[ */
		LAYER_KIND_GPU		= 0x02 << 16,	/**< GPU�������C���[ */

		LAYER_KIND_MEMORYTYPE		= 0x0F << 20,
		LAYER_KIND_UNKNOWNMEMORY	= 0x00 << 20,	
		LAYER_KIND_HOSTMEMORY		= 0x01 << 20,	/**< �z�X�g�������Ńf�[�^�̂��������s�� */
		LAYER_KIND_DEVICEMEMORY		= 0x02 << 20,	/**< �f�o�C�X�������Ńf�[�^�̂��������s�� */
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
		//=======================================
		// ���ʏ���
		//=======================================
		/** ���C���[��ʂ̎擾.
			ELayerKind �̑g�ݍ��킹. */
		virtual U32 GetLayerKind(void)const = 0;

		/** ���C���[�ŗL��GUID���擾���� */
		virtual Gravisbell::GUID GetGUID(void)const = 0;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::GUID GetLayerCode(void)const = 0;

		/** ���C���[�̐ݒ�����擾���� */
		virtual const SettingData::Standard::IData* GetLayerStructure()const = 0;

	public:
		//=======================================
		// ����������
		//=======================================
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		virtual ErrorCode Initialize(void) = 0;


	public:
		//=======================================
		// ���Z�O����
		//=======================================
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

		/** �o�b�`�T�C�Y���擾����.
			@return �����ɉ��Z���s���o�b�`�̃T�C�Y */
		virtual unsigned int GetBatchSize()const = 0;

	public:
		//=======================================
		// ���Z���[�v�O����
		//=======================================
		/** ���[�v�̏���������.�f�[�^�Z�b�g�̎��s�J�n�O�Ɏ��s����
			���s�����ꍇ��Calculate�ȍ~�̏����͎��s�s��. */
		virtual ErrorCode PreProcessLoop() = 0;



		//====================================
		// ���s���ݒ�
		//====================================
		/** ���s���ݒ���擾����. */
		virtual const SettingData::Standard::IData* GetRuntimeParameter()const = 0;
		virtual SettingData::Standard::IData* GetRuntimeParameter() = 0;

		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^�Aenum�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, S32 i_param) = 0;
		/** ���s���ݒ��ݒ肷��.
			int�^�Afloat�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, F32 i_param) = 0;
		/** ���s���ݒ��ݒ肷��.
			bool�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, bool i_param) = 0;
		/** ���s���ݒ��ݒ肷��.
			string�^���Ώ�.
			@param	i_dataID	�ݒ肷��l��ID.
			@param	i_param		�ݒ肷��l. */
		virtual ErrorCode SetRuntimeParameter(const wchar_t* i_dataID, const wchar_t* i_param) = 0;
	};

}	// Layer
}	// Gravisbell

#endif