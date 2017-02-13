//==================================
// �ݒ荀��(�񋓌^) ���������݉\
//==================================
#ifndef __I_NN_LAYER_CONFIG_EX_ENUM_H__
#define __I_NN_LAYER_CONFIG_EX_ENUM_H__

#include<INNLayerConfig.h>

namespace CustomDeepNNLibrary
{
	class INNLayerConfigItemEx_Enum : public INNLayerConfigItem_Enum
	{
	public:
		/** �R���X�g���N�^ */
		INNLayerConfigItemEx_Enum(){}
		/** �f�X�g���N�^ */
		virtual ~INNLayerConfigItemEx_Enum(){}


	public:
		/** �񋓒l��ǉ�����.
			����ID�����ɑ��݂���ꍇ���s����.
			@param szEnumID		�ǉ�����ID
			@param szEnumName	�ǉ����閼�O
			@param szComment	�ǉ�����R�����g��.
			@return ���������ꍇ�A�ǉ����ꂽ�A�C�e���̔ԍ����Ԃ�. ���s�����ꍇ�͕��̒l���Ԃ�. */
		virtual int AddEnumItem(const char szEnumID[], const char szEnumName[], const char szComment[]) = 0;

		/** �񋓒l���폜����.
			@param num	�폜����񋓔ԍ�
			@return ���������ꍇ0 */
		virtual int EraseEnumItem(int num) = 0;
		/** �񋓒l���폜����
			@param szEnumID �폜�����ID
			@return ���������ꍇ0 */
		virtual int EraseEnumItem(const char szEnumID[]) = 0;

		/** �f�t�H���g�l��ݒ肷��.	�ԍ��w��.
			@param num �f�t�H���g�l�ɐݒ肷��ԍ�. 
			@return ���������ꍇ0 */
		virtual int SetDefaultItem(int num) = 0;
		/** �f�t�H���g�l��ݒ肷��.	ID�w��. 
			@param szID �f�t�H���g�l�ɐݒ肷��ID. 
			@return ���������ꍇ0 */
		virtual int SetDefaultItem(const char szEnumID[]) = 0;

	};
}

#endif // __I_NN_LAYER_CONFIG_EX_ENUM_H__