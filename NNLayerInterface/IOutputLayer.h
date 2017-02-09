//=======================================
// レイヤーベース
//=======================================
#ifndef __I_OUTPUT_LAYER_H__
#define __I_OUTPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** レイヤーベース */
	class IOutputLayer : public virtual ILayerBase
	{
	public:
		/** コンストラクタ */
		IOutputLayer(){}
		/** デストラクタ */
		virtual ~IOutputLayer(){}

	public:
		/** 出力データ構造を取得する */
		virtual IODataStruct GetOutputDataStruct()const = 0;

		/** 出力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual unsigned int GetOutputBufferCount()const = 0;

		/** 出力データバッファを取得する.
			配列の要素数はGetOutputBufferCountの戻り値.
			@return 出力データ配列の先頭ポインタ */
		virtual const float* GetOutputBuffer()const = 0;
		/** 出力データバッファを取得する.
			@param lpOutputBuffer	出力データ格納先配列. GetOutputBufferCountで取得した値の要素数が必要
			@return 成功した場合0 */
		virtual ELayerErrorCode GetOutputBuffer(float lpOutputBuffer[])const = 0;

	public:
		/** 出力先レイヤーへのリンクを追加する.
			@param	pLayer	追加する出力先レイヤー
			@return	成功した場合0 */
		virtual ELayerErrorCode AddOutputToLayer(class IInputLayer* pLayer) = 0;
		/** 出力先レイヤーへのリンクを削除する.
			@param	pLayer	削除する出力先レイヤー
			@return	成功した場合0 */
		virtual ELayerErrorCode EraseOutputToLayer(class IInputLayer* pLayer) = 0;

	public:
		/** 出力先レイヤー数を取得する */
		virtual unsigned int GetOutputToLayerCount()const = 0;
		/** 出力先レイヤーのアドレスを番号指定で取得する.
			@param num	取得するレイヤーの番号.
			@return	成功した場合出力先レイヤーのアドレス.失敗した場合はNULLが返る. */
		virtual class IInputLayer* GetOutputToLayerByNum(unsigned int num)const = 0;
	};
}

#endif