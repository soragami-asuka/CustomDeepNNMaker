//=====================================
// レイヤー設定情報を作成する関数群
//=====================================
#ifndef __GRAVISBELL_NN_LAYER_CONFIG_H__
#define __GRAVISBELL_NN_LAYER_CONFIG_H__

#ifdef NNLAYERCONFIG_EXPORTS
#define NNLAYERCONFIG_API __declspec(dllexport)
#else
#define NNLAYERCONFIG_API __declspec(dllimport)
#endif

#include"Common/VersionCode.h"

#include"NNLayerInterface/ILayerConfig.h"
#include"ILayerConfigEx.h"
#include"ILayerConfigItemEx_Enum.h"

namespace Gravisbell {
namespace NeuralNetwork {

	/** 空のレイヤー設定情報を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigEx* CreateEmptyLayerConfig(const GUID& layerCode, const VersionCode& versionCode);

	/** 設定項目(実数)を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Float* CreateLayerCofigItem_Float(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], float minValue, float maxValue, float defaultValue);
	/** 設定項目(整数)を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Int* CreateLayerCofigItem_Int(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], int minValue, int maxValue, int defaultValue);
	/** 設定項目(文字列)を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_String* CreateLayerCofigItem_String(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], const wchar_t szDefaultValue[]);
	/** 設定項目(論理値)を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigItem_Bool* CreateLayerCofigItem_Bool(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[], bool defaultValue);
	/** 設定項目(列挙値)を作成する */
	extern "C" NNLAYERCONFIG_API ILayerConfigItemEx_Enum* CreateLayerCofigItem_Enum(const wchar_t i_szID[], const wchar_t i_szName[], const wchar_t i_szText[]);

}	// NeuralNetwork
}	// Gravisbell


#endif // __NN_LAYERCONFIG_H__