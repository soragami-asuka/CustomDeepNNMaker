//===========================
// NN�̃��C���[�ݒ荀�ڃt�H�[�}�b�g�x�[�X
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__

#include"SettingData/Standard/IData.h"

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class IDataEx : public IData
	{
	public:
		/** �R���X�g���N�^ */
		IDataEx()
			: IData()
		{
		}
		/** �f�X�g���N�^ */
		virtual ~IDataEx(){}

	public:
		/** �A�C�e����ǉ�����.
			�ǉ����ꂽ�A�C�e���͓�����delete�����. */
		virtual int AddItem(IItemBase* pItem)=0;

		/** �o�b�t�@����ǂݍ���.
			@param i_lpBuffer	�ǂݍ��݃o�b�t�@�̐擪�A�h���X.
			@param i_bufferSize	�ǂݍ��݉\�o�b�t�@�̃T�C�Y.
			@return	���ۂɓǂݎ�����o�b�t�@�T�C�Y. ���s�����ꍇ�͕��̒l */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};

}	// Standard
}	// SettingData
}	// Gravisbell

#endif // __I_NN_LAYER_CONFIG_WRITE_ABLE_H__