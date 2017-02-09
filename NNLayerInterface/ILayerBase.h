//=======================================
// レイヤーベース
//=======================================
#ifndef __I_LAYER_BASE_H__
#define __I_LAYER_BASE_H__

#include<guiddef.h>

#include"LayerErrorCode.h"
#include"IODataStruct.h"
#include"INNLayerConfig.h"

#ifndef BYTE
typedef unsigned char BYTE;
#endif

namespace CustomDeepNNLibrary
{
	/** レイヤー種別 */
	enum ELayerKind
	{
		LAYER_KIND_CPU = 0x00 << 8,	/**< CPU処理レイヤー */
		LAYER_KIND_GPU = 0x01 << 8,	/**< GPU処理レイヤー */

		LAYER_KIND_INPUT  = 0x00 << 0,	/**< 入力レイヤー */
		LAYER_KIND_OUTPUT = 0x01 << 0,	/**< 出力レイヤー */
		LAYER_KIND_CALC   = 0x02 << 0,	/**< 計算レイヤー,中間層 */

		LAYER_KIND_CPU_INPUT  = LAYER_KIND_CPU | LAYER_KIND_INPUT,
		LAYER_KIND_CPU_OUTPUT = LAYER_KIND_CPU | LAYER_KIND_OUTPUT,
		LAYER_KIND_CPU_CALC   = LAYER_KIND_CPU | LAYER_KIND_CALC,
		
		LAYER_KIND_GPU_INPUT  = LAYER_KIND_GPU | LAYER_KIND_INPUT,
		LAYER_KIND_GPU_OUTPUT = LAYER_KIND_GPU | LAYER_KIND_OUTPUT,
		LAYER_KIND_GPU_CALC   = LAYER_KIND_GPU | LAYER_KIND_CALC,
	};

	/** レイヤーベース */
	class ILayerBase
	{
	public:
		/** コンストラクタ */
		ILayerBase(){}
		/** デストラクタ */
		virtual ~ILayerBase(){}

	public:
		/** レイヤー種別の取得 */
		virtual ELayerKind GetLayerKind()const = 0;

		/** レイヤー固有のGUIDを取得する */
		virtual ELayerErrorCode GetGUID(GUID& o_guid)const = 0;
		
		/** レイヤー種別識別コードを取得する.
			@param o_layerCode	格納先バッファ
			@return 成功した場合0 */
		virtual ELayerErrorCode GetLayerCode(GUID& o_layerCode)const = 0;

	public:
		/** 演算前処理を実行する.
			@param batchSize	同時に演算を行うバッチのサイズ.
			NN作成後、演算処理を実行する前に一度だけ必ず実行すること。データごとに実行する必要はない.
			失敗した場合はCalculate以降の処理は実行不可. */
		virtual ELayerErrorCode PreCalculate(unsigned int batchSize) = 0;

		/** バッチサイズを取得する.
			@return 同時に演算を行うバッチのサイズ */
		virtual unsigned int GetBatchSize()const = 0;
	};
}

#endif