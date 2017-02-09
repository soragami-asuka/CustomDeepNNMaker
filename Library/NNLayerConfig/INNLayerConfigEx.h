//===========================
// NNのレイヤー設定項目フォーマットベース
//===========================
#ifndef __I_NN_LAYER_CONFIG_WRITE_ABLE_H__
#define __I_NN_LAYER_CONFIG_WRITE_ABLE_H__

#include<INNLayerConfig.h>

namespace CustomDeepNNLibrary
{
	class INNLayerConfigEx : public INNLayerConfig
	{
	public:
		/** コンストラクタ */
		INNLayerConfigEx()
			: INNLayerConfig()
		{
		}
		/** デストラクタ */
		virtual ~INNLayerConfigEx(){}

	public:
		/** アイテムを追加する.
			追加されたアイテムは内部でdeleteされる. */
		virtual int AddItem(INNLayerConfigItemBase* pItem)=0;

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};
}

#endif // __I_NN_LAYER_CONFIG_WRITE_ABLE_H__