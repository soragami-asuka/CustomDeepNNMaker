//======================================
// フィードフォワードニューラルネットワークの統合処理レイヤー
// 結合、活性化
// GPU処理用
//======================================
#include"stdafx.h"

#include"NNLayer_Feedforward.h"
#include"NNLayer_FeedforwardBase.h"





/** CPU処理用のレイヤーを作成 */
EXPORT_API CustomDeepNNLibrary::INNLayer* CreateLayerGPU(GUID guid)
{
//	return new NNLayer_FeedforwardCPU(guid);
	return NULL;
}