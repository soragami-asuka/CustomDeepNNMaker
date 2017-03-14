//===========================
// NNのレイヤー設定項目フォーマットベース
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__

#include"SettingData/Standard/IData.h"

namespace Gravisbell {
namespace SettingData {
namespace Standard {

	class IDataEx : public IData
	{
	public:
		/** コンストラクタ */
		IDataEx()
			: IData()
		{
		}
		/** デストラクタ */
		virtual ~IDataEx(){}

	public:
		/** アイテムを追加する.
			追加されたアイテムは内部でdeleteされる. */
		virtual int AddItem(IItemBase* pItem)=0;

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};

}	// Standard
}	// SettingData
}	// Gravisbell

#endif // __I_NN_LAYER_CONFIG_WRITE_ABLE_H__