//======================================
// �t�B�[�h�t�H���[�h�j���[�����l�b�g���[�N�̓����������C���[
// �����A������
// GPU�����p
//======================================
#include"stdafx.h"

#include"FullyConnect_Activation_DATA.hpp"
#include"FullyConnect_Activation_FUNC.hpp"
#include"FullyConnect_Activation_Base.h"

using namespace Gravisbell;
using namespace Gravisbell::Layer::NeuralNetwork;




/** CPU�����p�̃��C���[���쐬 */
EXPORT_API Gravisbell::Layer::NeuralNetwork::INNLayer* CreateLayerGPU(Gravisbell::GUID guid, const ILayerDLLManager* pLayerDLLManager)
{
//	return new FeedforwardCPU(guid);
	return NULL;
}