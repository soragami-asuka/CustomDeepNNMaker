//=======================================
// レイヤーDLLクラス
//=======================================
#ifndef __I_NN_LAYER_DLL_H__
#define __I_NN_LAYER_DLL_H__

#include<guiddef.h>

#include"INNLayer.h"

namespace CustomDeepNNLibrary
{
	class INNLayerDLL
	{
	public:
		/** コンストラクタ */
		INNLayerDLL(){}
		/** デストラクタ */
		virtual ~INNLayerDLL(){}

	public:
		/** レイヤー識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_layerCode)const = 0;
		/** バージョンコードを取得する.
			@param o_versionCode	格納先バッファ
			@return 成功した場合0 */
		virtual ELayerErrorCode GetVersionCode(CustomDeepNNLibrary::VersionCode& o_versionCode)const = 0;


		/** レイヤー設定を作成する */
		virtual INNLayerConfig* CreateLayerConfig(void)const = 0;
		/** レイヤー設定を作成する
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@param o_useBufferSize 実際に読み込んだバッファサイズ
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual INNLayerConfig* CreateLayerConfigFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize, int& o_useBufferSize)const = 0;

		
		/** CPU処理用のレイヤーを作成.
			GUIDは自動割り当て. */
		virtual INNLayer* CreateLayerCPU()const = 0;
		/** CPU処理用のレイヤーを作成
			@param guid	作成レイヤーのGUID */
		virtual INNLayer* CreateLayerCPU(GUID guid)const = 0;
		
		/** GPU処理用のレイヤーを作成.
			GUIDは自動割り当て. */
		virtual INNLayer* CreateLayerGPU()const = 0;
		/** GPU処理用のレイヤーを作成 */
		virtual INNLayer* CreateLayerGPU(GUID guid)const = 0;
	};
}

#endif