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

namespace Gravisbell {
namespace Layer {

	class ILayerData
	{
	public:
		/** �R���X�g���N�^ */
		ILayerData(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerData(){}

	public:
		/** ���C���[�ŗL��GUID���擾���� */
		virtual Gravisbell::GUID GetGUID(void)const = 0;

		/** ���C���[��ʎ��ʃR�[�h���擾����.
			@param o_layerCode	�i�[��o�b�t�@
			@return ���������ꍇ0 */
		virtual Gravisbell::GUID GetLayerCode(void)const = 0;


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
	};

}	// Layer
}	// Gravisbell

#endif