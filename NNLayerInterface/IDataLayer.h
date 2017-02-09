//=======================================
// データレイヤー
//=======================================
#ifndef __I_DATA_LAYER_H__
#define __I_DATA_LAYER_H__

#include"LayerErrorCode.h"
#include"IOutputLayer.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** 入力データレイヤー */
	class IDataLayer
	{
	public:
		/** コンストラクタ */
		IDataLayer(){}
		/** デストラクタ */
		virtual ~IDataLayer(){}

	public:
		/** データの構造情報を取得する */
		virtual IODataStruct GetDataStruct()const = 0;

		/** データのバッファサイズを取得する.
			@return データのバッファサイズ.使用するfloat型配列の要素数. */
		virtual unsigned int GetBufferCount()const = 0;

		/** データを追加する.
			@param	lpData	データ一組の配列. GetBufferSize()の戻り値の要素数が必要. */
		virtual ELayerErrorCode AddData(const float lpData[]) = 0;

		/** データ数を取得する */
		virtual unsigned int GetDataCount()const = 0;
		/** データを番号指定で取得する.
			@param num		取得する番号
			@param o_LpData データの格納先配列. GetBufferSize()の戻り値の要素数が必要. */
		virtual ELayerErrorCode GetDataByNum(unsigned int num, float o_lpData[])const = 0;
		/** 使用データの切り替え */
		virtual ELayerErrorCode ChangeUseDataByNum(unsigned int num) = 0;
		/** データを番号指定で消去する */
		virtual ELayerErrorCode EraseDataByNum(unsigned int num) = 0;

		/** データを全消去する.
			@return	成功した場合0 */
		virtual ELayerErrorCode ClearData() = 0;
	};
}

#endif