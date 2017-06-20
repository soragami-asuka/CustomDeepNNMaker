//=======================================
// ���C���[�Ɋւ���f�[�^����舵���C���^�[�t�F�[�X
// �o�b�t�@�Ȃǂ��Ǘ�����.
//=======================================
#ifndef __GRAVISBELL_I_LAYER_DATA_H__
#define __GRAVISBELL_I_LAYER_DATA_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

#include"./ILayerBase.h"

namespace Gravisbell {
namespace Layer {

	class ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerData(){}

		//===========================
		// ������
		//===========================
	public:
		/** ������. �e�j���[�����̒l�������_���ɏ�����
			@return	���������ꍇ0 */
		virtual ErrorCode Initialize(void) = 0;


		//===========================
		// ���ʐ���
		//===========================
	public:
		/** ���C���[�ŗL��GUID���擾���� */
		virtual Gravisbell::GUID GetGUID(void)const = 0;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::GUID GetLayerCode(void)const = 0;

		/** ���C���[�̐ݒ�����擾���� */
		virtual const SettingData::Standard::IData* GetLayerStructure()const = 0;


		//===========================
		// ���C���[�ۑ�
		//===========================
	public:
		/** ���C���[�̕ۑ��ɕK�v�ȃo�b�t�@����BYTE�P�ʂŎ擾���� */
		virtual U32 GetUseBufferByteCount()const = 0;

		/** ���C���[���o�b�t�@�ɏ�������.
			@param o_lpBuffer	�������ݐ�o�b�t�@�̐擪�A�h���X. GetUseBufferByteCount�̖߂�l�̃o�C�g�����K�v
			@return ���������ꍇ�������񂾃o�b�t�@�T�C�Y.���s�����ꍇ�͕��̒l */
		virtual S32 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
		
	public:
		//===========================
		// ���C���[�쐬
		//===========================
		/** ���C���[���쐬����.
			@param guid	�V�K�������郌�C���[��GUID. */
		virtual ILayerBase* CreateLayer(const Gravisbell::GUID& guid) = 0;

	public:
		//===========================
		// �I�v�e�B�}�C�U�[�ݒ�
		//===========================
		/** �I�v�e�B�}�C�U�[��ύX���� */
		virtual ErrorCode ChangeOptimizer(const wchar_t i_optimizerID[]) = 0;
		/** �I�v�e�B�}�C�U�[�̃n�C�p�[�p�����[�^��ύX���� */
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], F32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], S32 i_value) = 0;
		virtual ErrorCode SetOptimizerHyperParameter(const wchar_t i_parameterID[], const wchar_t i_value[]) = 0;
	};

}	// Layer
}	// Gravisbell

#endif