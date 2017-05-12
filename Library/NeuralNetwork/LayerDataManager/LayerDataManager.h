//=========================================
// �j���[�����l�b�g���[�N�p���C���[�f�[�^�̊Ǘ��N���X
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

	/** ���C���[�f�[�^�̊Ǘ��N���X���쐬 */
	LayerDataManager_API ILayerDataManager* CreateLayerDataManager();

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
