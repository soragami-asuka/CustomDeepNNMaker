//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_DATA.hpp"
#include"FullyConnect_Activation_FUNC.hpp"
#include"FullyConnect_Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;




/** CPU処理用のレイヤーを作成 */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid, const ILayerDLLManager* pLayerDLLManager)
{
//	return new FeedforwardCPU(guid);
	return NULL;
}