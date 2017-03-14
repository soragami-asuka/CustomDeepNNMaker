//=====================================
// レイヤー設定情報を作成する関数群
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

	/** 空のレイヤー設定情報を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IDataEx* CreateEmptyData(const GUID& layerCode, const VersionCode& versionCode);

	/** 設定項目(実数)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Float* CreateItem_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], float minValue, float maxValue, float defaultValue);
	/** 設定項目(整数)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Int* CreateItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], int minValue, int maxValue, int defaultValue);
	/** 設定項目(文字列)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_String* CreateItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[]);
	/** 設定項目(論理値)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItem_Bool* CreateItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue);
	/** 設定項目(列挙値)を作成する */
	extern "C" GRAVISBELL_SETTINGDATA_API IItemEx_Enum* CreateItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[]);

}	// Standard
}	// SettingData
}	// Gravisbell


#endif // __NN_LAYERCONFIG_H__