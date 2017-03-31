//=======================================
// ニューラルネットワーク本体定義
//=======================================
#ifndef __GRAVISBELL_I_NEURAL_NETWORK_H__
#define __GRAVISBELL_I_NEURAL_NETWORK_H__

#include"INNLayer.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	class INueralNetwork : public INNLayer
	{
	public:
		/** コンストラクタ */
		INueralNetwork(){}
		/** デストラクタ */
		virtual ~INueralNetwork(){}

	public:
		/** レイヤーを追加する.
			追加したレイヤーの所有権はNeuralNetworkに移るため、メモリの開放処理などは全てINeuralNetwork内で行われる.
			@param pLayer	追加するレイヤーのアドレス. */
		virtual ErrorCode AddLayer(INNLayer* pLayer) = 0;
		/** レイヤーを削除する.
			@param guid	削除するレイヤーのGUID */
		virtual ErrorCode EraseLayer(const Gravisbell::GUID& guid) = 0;
		/** レイヤーを全削除する */
		virtual ErrorCode EraseAllLayer() = 0;

		/** 登録されているレイヤー数を取得する */
		virtual ErrorCode GetLayerCount()const = 0;
		/** レイヤーをGUID指定で取得する */
		virtual const INNLayer* GetLayerByGUID(const Gravisbell::GUID& guid) = 0;

		/** 入力信号に割り当てられているGUIDを取得する */
		virtual GUID GetInputGUID()const;
		/** 出力信号に割り当てらているレイヤーのGUIDを取得する */
		virtual GUID GetOutputLayerGUID()const;
	};

}	// NeuralNetwork
}	// Layer
}	// Gravisbell


#endif // __GRAVISBELL_I_NEURAL_NETWORK_H__
