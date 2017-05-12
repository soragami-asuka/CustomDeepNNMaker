//=========================================
// ニューラルネットワーク用レイヤーデータの管理クラス
//=========================================
#ifdef LAYERDATAMANAGER_EXPORTS
#define LayerDataManager_API __declspec(dllexport)
#else
#define LayerDataManager_API __declspec(dllimport)
#endif

#include"Common/ErrorCode.h"
#include"Common/VersionCode.h"

#include"Layer/NeuralNetwork/ILayerDataManager.h"


namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	/** レイヤーデータの管理クラスを作成 */
	LayerDataManager_API ILayerDataManager* CreateLayerDataManager();

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
