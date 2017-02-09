//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化を処理する
//======================================
#define EXPORT_API extern "C" __declspec(dllexport)


#include"LayerErrorCode.h"
#include"INNLayerConfig.h"
#include"INNLayer.h"

#include<guiddef.h>

/** レイヤー識別コードを取得する.
	@param o_layerCode	格納先バッファ
	@return 成功した場合0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetLayerCode(GUID& o_layerCode);

/** バージョンコードを取得する.
	@param o_versionCode	格納先バッファ
	@return 成功した場合0 */
EXPORT_API CustomDeepNNLibrary::ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode);


/** レイヤー設定を作成する */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfig(void);
/** レイヤー設定を作成する
	@param i_lpBuffer	読み込みバッファの先頭アドレス.
	@param i_bufferSize	読み込み可能バッファのサイズ.
	@param o_useBufferSize 実際に読み込んだバッファサイズ
	@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
EXPORT_API CustomDeepNNLibrary::INNLayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize);


/** CPU処理用のレイヤーを作成.
	@param guid	作成するレイヤーのGUID */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerCPU(GUID guid);

/** GPU処理用のレイヤーを作成
	@param guid 作成するレイヤーのGUID */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerGPU(GUID guid);
