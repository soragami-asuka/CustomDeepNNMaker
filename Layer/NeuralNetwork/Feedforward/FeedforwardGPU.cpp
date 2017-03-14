//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"Feedforward_DATA.hpp"
#include"FeedforwardBase.h"
#include"Feedforward_FUNC.hpp"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;




/** CPU処理用のレイヤーを作成 */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid)
{
//	return new FeedforwardCPU(guid);
	return NULL;
}