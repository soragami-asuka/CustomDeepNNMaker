//==================================
// �ݒ荀��(�񋓌^) ���������݉\
//==================================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_EX_ENUM_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_EX_ENUM_H__

#include"NNLayerInterface/ILayerConfig.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerConfigItemEx_Enum : public ILayerConfigItem_Enum
	{
	public:
		/** �R���X�g���N�^ */
		ILayerConfigItemEx_Enum(){}
		/** �f�X�g���N�^ */
		virtual ~ILayerConfigItemEx_Enum(){}


	public:
		/** �񋓒l��ǉ�����.
			����ID�����ɑ��݂���ꍇ���s����.
			@param szEnumID		�ǉ�����ID
			@param szEnumName	�ǉ����閼�O
			@param szComment	�ǉ�����R�����g��.
			@return ���������ꍇ�A�ǉ����ꂽ�A�C�e���̔ԍ����Ԃ�. ���s�����ꍇ�͕��̒l���Ԃ�. */
		virtual S32 AddEnumItem(const wchar_t szEnumID[], const wchar_t szEnumName[], const wchar_t szComment[]) = 0;

		/** �񋓒l���폜����.
			@param num	�폜����񋓔ԍ�
			@return ���������ꍇ0 */
		virtual S32 EraseEnumItem(S32 num) = 0;
		/** �񋓒l���폜����
			@param szEnumID �폜�����ID
			@return ���������ꍇ0 */
		virtual S32 EraseEnumItem(const wchar_t szEnumID[]) = 0;

		/** �f�t�H���g�l��ݒ肷��.	�ԍ��w��.
			@param num �f�t�H���g�l�ɐݒ肷��ԍ�. 
			@return ���������ꍇ0 */
		virtual S32 SetDefaultItem(S32 num) = 0;
		/** �f�t�H���g�l��ݒ肷��.	ID�w��. 
			@param szID �f�t�H���g�l�ɐݒ肷��ID. 
			@return ���������ꍇ0 */
		virtual S32 SetDefaultItem(const wchar_t szEnumID[]) = 0;

	};

}	// NeuralNetwork
}	// Gravisbell

#endif // __I_NN_LAYER_CONFIG_EX_ENUM_H__