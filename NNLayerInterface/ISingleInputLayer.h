//=======================================
// 単一入力レイヤー
//=======================================
#ifndef __I_SINGLE_INPUT_LAYER_H__
#define __I_SINGLE_INPUT_LAYER_H__

#include"LayerErrorCode.h"
#include"ILayerBase.h"
#include"IODataStruct.h"
#include"IInputLayer.h"

namespace CustomDeepNNLibrary
{
	/** レイヤーベース */
	class ISingleInputLayer : public virtual IInputLayer
	{
	public:
		/** コンストラクタ */
		ISingleInputLayer(){}
		/** デストラクタ */
		virtual ~ISingleInputLayer(){}

	public:
		/** 入力バッファ数を取得する. byte数では無くデータの数なので注意 */
		virtual unsigned int GetInputBufferCount()const = 0;

		/** 学習差分を取得する.
			配列の要素数は[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]
			@return	誤差差分配列の先頭ポインタ */
		virtual const float** GetDInputBuffer()const = 0;
		/** 学習差分を取得する.
			@param lpDOutputBuffer	学習差分を格納する配列.[GetBatchSize()の戻り値][GetInputBufferCount()の戻り値]の配列が必要 */
		virtual ELayerErrorCode GetDInputBuffer(float** o_lpDInputBuffer)const = 0;

	public:
		/** 入力データ構造を取得する.
			@return	入力データ構造 */
		virtual const IODataStruct GetInputDataStruct()const = 0;
		/** 入力データ構造を取得する
			@param	o_inputDataStruct	入力データ構造の格納先
			@return	成功した場合0 */
		virtual ELayerErrorCode GetInputDataStruct(IODataStruct& o_inputDataStruct)const = 0;
	};
}

#endif