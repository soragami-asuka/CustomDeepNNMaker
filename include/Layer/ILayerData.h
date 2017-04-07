//=======================================
// レイヤーに関するデータを取り扱うインターフェース
// バッファなどを管理する.
//=======================================
#ifndef __GRAVISBELL_I_LAYER_DATA_H__
#define __GRAVISBELL_I_LAYER_DATA_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"

#include"../SettingData/Standard/IData.h"

namespace Gravisbell {
namespace Layer {

	class ILayerData
	{
	public:
		/** コンストラクタ */
		ILayerData(){}
		/** デストラクタ */
		virtual ~ILayerData(){}

	public:
		/** レイヤー固有のGUIDを取得する */
		virtual Gravisbell::GUID GetGUID(void)const = 0;

		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		virtual Gravisbell::GUID GetLayerCode(void)const = 0;


		//===========================
		// レイヤー保存
		//===========================
	public:
		/** レイヤーの保存に必要なバッファ数をBYTE単位で取得する */
		virtual U32 GetUseBufferByteCount()const = 0;

		/** レイヤーをバッファに書き込む.
			@param o_lpBuffer	書き込み先バッファの先頭アドレス. GetUseBufferByteCountの戻り値のバイト数が必要
			@return 成功した場合書き込んだバッファサイズ.失敗した場合は負の値 */
		virtual S32 WriteToBuffer(BYTE* o_lpBuffer)const = 0;
	};

}	// Layer
}	// Gravisbell

#endif