//===========================
// NNのレイヤー設定項目フォーマットベース
//===========================
#ifndef __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__
#define __GRAVISBELL_I_NN_LAYER_CONFIG_WRITE_ABLE_H__

#include"NNLayerInterface/ILayerConfig.h"

namespace Gravisbell {
namespace NeuralNetwork {

	class ILayerConfigEx : public ILayerConfig
	{
	public:
		/** コンストラクタ */
		ILayerConfigEx()
			: ILayerConfig()
		{
		}
		/** デストラクタ */
		virtual ~ILayerConfigEx(){}

	public:
		/** アイテムを追加する.
			追加されたアイテムは内部でdeleteされる. */
		virtual int AddItem(ILayerConfigItemBase* pItem)=0;

		/** バッファから読み込む.
			@param i_lpBuffer	読み込みバッファの先頭アドレス.
			@param i_bufferSize	読み込み可能バッファのサイズ.
			@return	実際に読み取ったバッファサイズ. 失敗した場合は負の値 */
		virtual int ReadFromBuffer(const BYTE* i_lpBuffer, int i_bufferSize) = 0;
	};

}	// NeuralNetwork
}	// Gravisbell

#endif // __I_NN_LAYER_CONFIG_WRITE_ABLE_H__