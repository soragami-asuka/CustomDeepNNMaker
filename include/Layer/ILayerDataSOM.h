//=======================================
// SOMレイヤーを取り扱うレイヤーデータ
//=======================================
#ifndef __GRAVISBELL_I_LAYER_DATA_SOM_H__
#define __GRAVISBELL_I_LAYER_DATA_SOM_H__

#include"../Common/Common.h"
#include"../Common/ErrorCode.h"
#include"../Common/IODataStruct.h"
#include"../Common/Guiddef.h"
#include"../Common/ITemporaryMemoryManager.h"

#include"../SettingData/Standard/IData.h"

#include"./ILayerBase.h"
#include"./ILayerData.h"

namespace Gravisbell {
namespace Layer {

	class ILayerDataSOM : public ILayerData
	{
	public:
		/** コンストラクタ */
		ILayerDataSOM() : ILayerData(){}
		/** デストラクタ */
		virtual ~ILayerDataSOM(){}


		//==================================
		// SOM関連処理
		//==================================
	public:
		/** マップサイズを取得する.
			@return	マップのバッファ数を返す.(F32配列の要素数) */
		virtual U32 GetMapSize()const = 0;

		/** マップのバッファを取得する.
			@param	o_lpMapBuffer	マップを格納するホストメモリバッファ. GetMapSize()の戻り値の要素数が必要. */
		virtual Gravisbell::ErrorCode GetMapBuffer(F32* o_lpMapBuffer)const = 0;
	};

}	// Layer
}	// Gravisbell

#endif