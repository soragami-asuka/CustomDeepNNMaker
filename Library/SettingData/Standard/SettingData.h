//=====================================
// ���C���[�ݒ�����쐬����֐��Q
//=====================================
#ifndef __GRAVISBELL_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_NN_LAYER_CONFIG_H__

#ifdef Standard_EXPORTS
#define GRAVISBELL_SETTINGDATA_API __declspec(dllexport)
#else
#define GRAVISBELL_SETTINGDATA_API __declspec(dllimport)
#endif

#include"Common/VersionCode.h"
#include"SettingData/Standard/IData.h"

#include"IDataEx.h"
#include"IItemEx_Enum.h"

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	/** ��̃��C���[�ݒ�����쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IDataEx* CreateEmptyData(const GUID& layerCode, const VersionCode& versionCode);

	/** �ݒ荀��(����)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Float* CreateItem_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], float minValue, float maxValue, float defaultValue);
	/** �ݒ荀��(����)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Int* CreateItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], int minValue, int maxValue, int defaultValue);
	/** �ݒ荀��(������)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_String* CreateItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[]);
	/** �ݒ荀��(�_���l)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Bool* CreateItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue);
	/** �ݒ荀��(�񋓒l)���쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IItemEx_Enum* CreateItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[]);

}	// Standard
}	// SettingData
}	// Gravisbell


#endif // __NN_LAYERCONFIG_H__