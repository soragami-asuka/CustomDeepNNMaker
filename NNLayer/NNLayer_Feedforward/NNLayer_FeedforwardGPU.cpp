//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward_DATA.hpp"
#include"NNLayer_FeedforwardBase.h"
#include"NNLayer_Feedforward_FUNC.hpp"

using namespace Gravisbell;
using namespace Gravisbell::NeuralNetwork;




/** CPU処理用のレイヤーを作成 */
EXPORT_API Gravisbell::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid)
{
//	return new NNLayer_FeedforwardCPU(guid);
	return NULL;
}