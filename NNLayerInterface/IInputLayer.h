//=======================================
// レイヤーベース
//=======================================
#ifndef __I_INPUT_LAYER_H__
#define __I_INPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"

namespace CustomDeepNNLibrary
{
	/** レイヤーベース */
	class IInputLayer : public virtual ILayerBase
	{
	public:
		/** コンストラクタ */
		IInputLayer(){}
		/** デストラクタ */
		virtual ~IInputLayer(){}

	public:
		/** 学習誤差を計算する.
			直前の計算結果を使用する */
		virtual ELayerErrorCode CalculateLearnError() = 0;

	public:
		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual unsigned int GetInputBufferCount()const = 0;

		/** 学習差分を取得する.
			配列の要素数はGetInputBufferCountの戻り値.
			@return	誤差差分配列の先頭ポインタ */
		virtual const float* GetDInputBuffer()const = 0;
		/** 学習差分を取得する.
			@param lpDOutputBuffer	学習差分を格納する配列. GetInputBufferCountで取得した値の要素数が必要 */
		virtual ELayerErrorCode GetDInputBuffer(float o_lpDInputBuffer[])const = 0;

	public:
		/** 入力元レイヤーへのリンクを追加する.
			@param	pLayer	追加する入力元レイヤー
			@return	成功した場合0 */
		virtual ELayerErrorCode AddInputFromLayer(class IOutputLayer* pLayer) = 0;
		/** 入力元レイヤーへのリンクを削除する.
			@param	pLayer	削除する入力元レイヤー
			@return	成功した場合0 */
		virtual ELayerErrorCode EraseInputFromLayer(class IOutputLayer* pLayer) = 0;

	public:
		/** 入力元レイヤー数を取得する */
		virtual unsigned int GetInputFromLayerCount()const = 0;
		/** 入力元レイヤーのアドレスを番号指定で取得する.
			@param num	取得するレイヤーの番号.
			@return	成功した場合入力元レイヤーのアドレス.失敗した場合はNULLが返る. */
		virtual class IOutputLayer* GetInputFromLayerByNum(unsigned int num)const = 0;

		/** 入力元レイヤーが入力バッファのどの位置に居るかを返す.
			※対象入力レイヤーの前にいくつの入力バッファが存在するか.
			　学習差分の使用開始位置としても使用する.
			@return 失敗した場合負の値が返る*/
		virtual int GetInputBufferPositionByLayer(const class IOutputLayer* pLayer) = 0;
	};
}

#endif