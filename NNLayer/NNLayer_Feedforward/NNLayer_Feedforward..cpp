// NNLayer_Feedforward.cpp : DLL アプリケーション用にエクスポートされる関数を定義します。
//

#include "stdafx.h"
#include "NNLayer_Feedforward.h"

#include"LayerErrorCode.h"
#include"NNLayerConfig.h"


/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetLayerCode(GUID& o_layerCode)
{
	// {BEBA34EC-C30C-4565-9386-56088981D2D7}
	static const GUID layerCode = { 0xbeba34ec, 0xc30c, 0x4565, { 0x93, 0x86, 0x56, 0x8, 0x89, 0x81, 0xd2, 0xd7 } };

	o_layerCode = layerCode;

	return CustomDeepNNLibrary::LAYER_ERROR_NONE;
}
/** バージョンコードを取得する.
	@param o_versionCode	格納先バッファ
	@return 成功した場合0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode)
{
	static const CustomDeepNNLibrary::VersionCode versionCode(0, 0, 0, 0);

	o_versionCode = versionCode;

	return CustomDeepNNLibrary::LAYER_ERROR_NONE;
}


/** レイヤー設定を作成する */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfig(void)
{
	GUID layerCode;
	GetLayerCode(layerCode);

	CustomDeepNNLibrary::VersionCode versionCode;
	GetVersionCode(versionCode);


	//=================================
	// 空の設定情報を作成する
	//=================================
	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = CustomDeepNNLibrary::CreateEmptyLayerConfig(layerCode, versionCode);
	if(pLayerConfig == NULL)
		return NULL;


	//=================================
	// 各設定アイテムを追加していく
	//=================================
	// ニューロン数
	{
		CustomDeepNNLibrary::INNLayerConfigItem_Int* pItem = CustomDeepNNLibrary::CreateLayerCofigItem_Int("ニューロン数", 0, 65535, 200);

		pLayerConfig->AddItem(pItem);
	}


	return pLayerConfig;
}

/** レイヤー設定を作成する
	@param i_lpBuffer	読み込みバッファの先頭アドレス.
	@param i_bufferSize	読み込み可能バッファのサイズ.
	@param o_useBufferSize 実際に読み込んだバッファサイズ
	@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)
{
	CustomDeepNNLibrary::INNLayerConfigEx* pLayerConfig = (CustomDeepNNLibrary::INNLayerConfigEx*)CreateLayerConfig();
	if(pLayerConfig == NULL)
		return NULL;

	int useBufferSize = pLayerConfig->ReadFromBuffer(i_lpBuffer, i_bufferSize);
	if(useBufferSize < 0)
	{
		delete pLayerConfig;

		return NULL;
	}
	o_useBufferSize = useBufferSize;

	return pLayerConfig;
}
