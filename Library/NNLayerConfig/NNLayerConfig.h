//=====================================
// レイヤー設定情報を作成する関数群
//=====================================
#ifndef __NN_LAYER_CONFIG_H__
#define __NN_LAYER_CONFIG_H__

#ifdef NNLAYERCONFIG_EXPORTS
#define NNLAYERCONFIG_API __declspec(dllexport)
#else
#define NNLAYERCONFIG_API __declspec(dllimport)
#endif

#include"INNLayerConfig.h"
#include"INNLayerConfigEx.h"
#include"INNLayerConfigItemEx_Enum.h"

namespace CustomDeepNNLibrary
{
	/** 空のレイヤー設定情報を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigEx* CreateEmptyLayerConfig(const GUID& layerCode, const VersionCode& versionCode);

	/** 設定項目(実数)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Float* CreateLayerCofigItem_Float(const char szName[], float minValue, float maxValue, float defaultValue);
	/** 設定項目(整数)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Int* CreateLayerCofigItem_Int(const char szName[], int minValue, int maxValue, int defaultValue);
	/** 設定項目(文字列)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_String* CreateLayerCofigItem_String(const char szName[], const char szDefaultValue[]);
	/** 設定項目(論理値)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItem_Bool* CreateLayerCofigItem_Bool(const char szName[], bool defaultValue);
	/** 設定項目(列挙値)を作成する */
	extern "C" NNLAYERCONFIG_API INNLayerConfigItemEx_Enum* CreateLayerCofigItem_Enum(const char szName[]);
}


#endif // __NN_LAYERCONFIG_H__