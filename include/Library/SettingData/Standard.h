//=====================================
// ���C���[�ݒ�����쐬����֐��Q
//=====================================
#ifndef __GRAVISBELL_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_NN_LAYER_CONFIG_H__

#ifdef Standard_EXPORTS
#define GRAVISBELL_SETTINGDATA_API __declspec(dllexport)
#else
#define GRAVISBELL_SETTINGDATA_API __declspec(dllimport)
#ifndef GRAVISBELL_LIBRARY
#pragma comment(lib, "Gravisbell.SettingData.Standard.lib")
#endif
#endif

#include"Common/VersionCode.h"
#include"SettingData/Standard/IData.h"

#include"../../SettingData/Standard/IDataEx.h"
#include"../../SettingData/Standard/IItemEx_Enum.h"


namespace Gravisbell {
namespace SettingData {
namespace Standard {

	/** ��̃��C���[�ݒ�����쐬���� */
	extern "C" GRAVISBELL_SETTINGDATA_API IDataEx* CreateEmptyData(const Gravisbell::GUID& layerCode, const VersionCode& versionCode);

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

	/** �ݒ荀��(Vector3)(����)���쐬����. */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Vector3D_Float* CreateItem_Vector3D_Float(
		const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[],
		F32 i_minValueX, F32 i_minValueY, F32 i_minValueZ,
		F32 i_maxValueX, F32 i_maxValueY, F32 i_maxValueZ,
		F32 i_defaultValueX, F32 i_defaultValueY, F32 i_defaultValueZ);

	/** �ݒ荀��(Vector3)(����)���쐬����. */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Vector3D_Int* CreateItem_Vector3D_Int(
		const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[],
		S32 i_minValueX, S32 i_minValueY, S32 i_minValueZ,
		S32 i_maxValueX, S32 i_maxValueY, S32 i_maxValueZ,
		S32 i_defaultValueX, S32 i_defaultValueY, S32 i_defaultValueZ);

}	// Standard
}	// SettingData
}	// Gravisbell


#endif // __NN_LAYERCONFIG_H__