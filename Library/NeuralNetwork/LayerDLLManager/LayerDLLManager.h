// �ȉ��� ifdef �u���b�N�� DLL ����̃G�N�X�|�[�g��e�Ղɂ���}�N�����쐬���邽�߂� 
// ��ʓI�ȕ��@�ł��B���� DLL ���̂��ׂẴt�@�C���́A�R�}���h ���C���Œ�`���ꂽ LayerDLLManager_EXPORTS
// �V���{�����g�p���ăR���p�C������܂��B���̃V���{���́A���� DLL ���g�p����v���W�F�N�g�ł͒�`�ł��܂���B
// �\�[�X�t�@�C�������̃t�@�C�����܂�ł��鑼�̃v���W�F�N�g�́A 
// LayerDLLManager_API �֐��� DLL ����C���|�[�g���ꂽ�ƌ��Ȃ��̂ɑ΂��A���� DLL �́A���̃}�N���Œ�`���ꂽ
// �V���{�����G�N�X�|�[�g���ꂽ�ƌ��Ȃ��܂��B
#ifdef LayerDLLManager_EXPORTS
#define LayerDLLManager_API __declspec(dllexport)
#else
#define LayerDLLManager_API __declspec(dllimport)
#endif

#include"Common/ErrorCode.h"
#include"Common/VersionCode.h"

#include"Layer/NeuralNetwork/ILayerDLLManager.h"

namespace Gravisbell {
namespace Layer {
namespace NeuralNetwork {

	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerCPU();
	extern LayerDLLManager_API ILayerDLLManager* CreateLayerDLLManagerGPU();

}	// NeuralNetwork
}	// Layer
}	// Gravisbell
